package ch.unibas.dmi.dbis.cottontail.database.schema

import ch.unibas.dmi.dbis.cottontail.database.general.DBO
import ch.unibas.dmi.dbis.cottontail.database.general.Transaction
import ch.unibas.dmi.dbis.cottontail.database.general.TransactionStatus
import ch.unibas.dmi.dbis.cottontail.model.exceptions.DatabaseException
import ch.unibas.dmi.dbis.cottontail.model.exceptions.TransactionException

import org.mapdb.*
import org.mapdb.volume.MappedFileVol

import java.nio.file.Files
import java.nio.file.Path
import java.util.*
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write

/**
 * Represents a single column in the Cottontail DB schema. A [Column] record is identified by a tuple
 * ID (long) and can hold an arbitrary value.
 *
 * @param <T> Type of the value held by this [Column].
 *
 * @author Ralph Gasser
 * @version 1.0
 */
internal class Column<T: Any>(override val name: String, entity: Entity): DBO {

    /** The [Path] to the [Entity]'s main folder. */
    override val path: Path = entity.path.resolve("col_$name.db")

    /** The fully qualified name of this [Column] */
    override val fqn: String = "${entity.parent!!.name}.${entity.name}.$name"

    /** The parent [DBO], which is the [Entity] in case of an [Column]. */
    override val parent: Entity? = entity

    /** Internal reference to the [Store] underpinning this [Column]. */
    private var store: StoreWAL = try {
        StoreWAL.make(file = this.path.toString(), volumeFactory = MappedFileVol.FACTORY, fileLockWait = this.parent!!.parent!!.config.lockTimeout)
    } catch (e: DBException) {
        throw DatabaseException("Failed to open column at '$path': ${e.message}'")
    }

    /** Internal reference to the [Header] of this [Column]. */
    private val header
        get() = store.get(HEADER_RECORD_ID, ColumnHeaderSerializer) ?: throw DatabaseException.DataCorruptionException("Failed to open header of column '$fqn'!'")

    /**
     * Getter for [Column.definition].
     *
     * @return The [ColumnType] of this [Column].
     */
    val type: ColumnType<T>
        get() = this.header.type as ColumnType<T>

    /**
     * Status indicating whether this [Column] is open or closed.
     */
    @Volatile
    override var closed: Boolean = false
        private set

    /** A internal lock that is used to synchronize [Column.Tx]s affecting this [Column]. */
    private val txLock = ReentrantReadWriteLock()

    /** A internal lock that is used to synchronize closing of an [Column] with running [Column.Tx]. */
    private val globalLock = ReentrantReadWriteLock()

    /**
     * Closes the [Column]. Closing an [Column] is a delicate matter since ongoing [Column.Tx]  are involved.
     * Therefore, access to the method is mediated by an global [Column] wide lock.
     */
    override fun close() = this.globalLock.write {
        this.closed = true
        this.store.close()
    }

    /**
     * Companion object with some important constants.
     */
    companion object {
        /** Record ID of the [ColumnHeader]. */
        private const val HEADER_RECORD_ID: Long = 1L

        /** The identifier that is used to identify a Cottontail DB [Column] file. */
        private const val HEADER_IDENTIFIER: String = "COTTONC"

        /** The version of the Cottontail DB [Column] file. */
        private const val HEADER_VERSION: Short = 1

        /**
         * Initializes a new, empty [Column]
         *
         * @param parent The folder that contains the data file.
         * @param definition The [ColumnDef] that specified the [Column]
         */
        fun initialize(definition: ColumnDef, path: Path) {
            val store = StoreWAL.make(file = path.resolve("col_${definition.name}.db").toString(), volumeFactory = MappedFileVol.FACTORY)
            store.put(ColumnHeader(type = definition.type, size = definition.size, nullable = definition.nullable), ColumnHeaderSerializer)
            store.commit()
            store.close()
        }
    }

    /**
     * A [Transaction] that affects this [Column].
     */
    inner class Tx(override val readonly: Boolean, override val tid: UUID = UUID.randomUUID()): Transaction {
        /** Flag indicating whether or not this [Entity.Tx] was closed */
        @Volatile override var status: TransactionStatus = TransactionStatus.CLEAN
            private set

        /** Tries to acquire a global read-lock on the [Column]. */
        init {
            if (this@Column.closed) {
                throw TransactionException.TransactionDBOClosedException(tid)
            }
            this@Column.globalLock.readLock().lock()
        }

        /**
         * Commits all changes made through this [Tx] since the last commit or rollback.
         */
        @Synchronized
        override fun commit() {
            if (this.status == TransactionStatus.DIRTY) {
                this@Column.store.commit()
                this.status = TransactionStatus.CLEAN
                this@Column.txLock.writeLock().unlock()
            }
        }

        /**
         * Rolls all changes made through this [Tx] back to the last commit. Can only be executed, if [Tx] is
         * in status [TransactionStatus.DIRTY] or [TransactionStatus.ERROR].
         */
        @Synchronized
        override fun rollback() {
            if (this.status == TransactionStatus.DIRTY || this.status == TransactionStatus.ERROR) {
                this@Column.store.rollback()
                this.status = TransactionStatus.CLEAN
                this@Column.txLock.writeLock().unlock()
            }
        }

        /**
         * Closes this [Tx] and relinquishes the associated [ReentrantReadWriteLock].
         */
        @Synchronized
        override fun close() {
            if (this.status == TransactionStatus.DIRTY || this.status == TransactionStatus.ERROR) {
                this@Column.store.rollback()
                this@Column.txLock.writeLock().unlock()
            }
            this.status = TransactionStatus.CLOSED
            this@Column.globalLock.readLock().unlock()
        }

        /**
         * Gets and returns an entry from this [Column]. Action acquires a global read dataLock for the [Column].
         *
         * @param tupleId The ID of the desired entry
         * @return The desired entry.
         *
         * @throws DatabaseException If the tuple with the desired ID doesn't exist OR is invalid.
         */
        fun read(tupleId: Long) : T? = this@Column.txLock.read {
            checkValidOrThrow()
            checkValidTupleId(tupleId)
            return this@Column.store.get(tupleId, this@Column.type.serializer)
        }

        /**
         * Returns the number of entries in this [Column]. Action acquires a global read dataLock for the [Column].
         *
         * @return The number of entries in this [Column].
         */
        fun count(): Long = this@Column.txLock.read {
            checkValidOrThrow()
            return this@Column.header.count
        }

        /**
         * Applies the provided mapping function on each value found in this [Column], returning a
         * collection of the desired output values.
         *
         * @param action The action that should be applied.
         * @return A collection of Pairs mapping the tupleId to the generated value.
         */
        fun <R> map(action: (T?) -> R?): Collection<Pair<Long,R?>> = this@Column.txLock.read {
            checkValidOrThrow()
            val list = mutableListOf<Pair<Long,R?>>()
            this@Column.store.getAllRecids().forEach {
                list.add(Pair(it,action(this.read(it))))
            }
            return list
        }

        /**
         * Applies the provided function on each element found in this [Column].
         */
        fun forEach(action: (Long,T) -> Unit) = this@Column.txLock.read {
            checkValidOrThrow()
            this@Column.store.getAllRecids().forEach {
                if (it != HEADER_RECORD_ID) {
                    action(it,this.read(it)!!)
                }
            }
        }

        /**
         * Inserts a new record in this [Column]. This action will set this [Column.Tx] to [TransactionStatus.DIRTY]
         * and acquire a column-wide write lock until the [Column.Tx] either commit or rollback is issued.
         *
         * @param record The record that should be inserted. Can be null!
         * @return The tupleId of the inserted record OR the allocated space in case of a null value.
         */
        fun insert(record: T?): Long = try {
            acquireWriteLock()
            val tupleId = if (record == null) {
                this@Column.store.preallocate()
            } else {
                this@Column.store.put(record, this@Column.type.serializer)
            }

            /* Update header. */
            val header = this@Column.header
            header.count += 1
            header.modified = System.currentTimeMillis()
            store.update(HEADER_RECORD_ID, header, ColumnHeaderSerializer)
            tupleId
        } catch (e: DBException) {
            this.status = TransactionStatus.ERROR
            throw TransactionException.TransactionStorageException(this.tid, e.message ?: "Unknown")
        }

        /**
         * Inserts a list of new records in this [Column]. This action will set this [Column.Tx] to [TransactionStatus.DIRTY]
         * and acquire a column-wide write lock until the [Column.Tx] either commit or rollback is issued.
         *
         * @param records The records that should be inserted. Can contain null values!
         * @return The tupleId of the inserted record OR the allocated space in case of a null value.
         */
        fun insertAll(records: Collection<T?>): Collection<Long> = try {
            acquireWriteLock()
            val tupleIds = records.map {
                if (it == null) {
                this@Column.store.preallocate()
            } else {
                this@Column.store.put(it, this@Column.type.serializer)
            } }

            /* Update header. */
            val header = this@Column.header
            header.count += records.size
            header.modified = System.currentTimeMillis()
            store.update(HEADER_RECORD_ID, header, ColumnHeaderSerializer)
            tupleIds
        } catch (e: DBException) {
            this.status = TransactionStatus.ERROR
            throw TransactionException.TransactionStorageException(this.tid, e.message ?: "Unknown")
        }

        /**
         * Updates the entry with the specified tuple ID and sets it to the new value. This action will set this [Column.Tx]
         * to [TransactionStatus.DIRTY] and acquire a column-wide write lock until the [Column.Tx] either commit or rollback is issued.
         *
         * @param tupleId The ID of the record that should be updated
         * @param value The new value.
         */
        fun update(tupleId: Long, value: T) = try {
            acquireWriteLock()
            checkValidTupleId(tupleId)
            this@Column.store.update(tupleId, value, this@Column.type.serializer)
        } catch (e: DBException) {
            this.status = TransactionStatus.ERROR
            throw TransactionException.TransactionStorageException(this.tid, e.message ?: "Unknown")
        }

        /**
         * Updates the entry with the specified tuple ID and sets it to the new value. This action will set this [Column.Tx]
         * to [TransactionStatus.DIRTY] and acquire a column-wide write lock until the [Column.Tx] either commit or rollback is issued.
         *
         * @param tupleId The ID of the record that should be updated
         * @param value The new value.
         * @param expected The value expected to be there.
         */
        fun compareAndUpdate(tupleId: Long, value: T, expected: T): Boolean = try {
            acquireWriteLock()
            checkValidTupleId(tupleId)
            this@Column.store.compareAndSwap(tupleId, expected, value, this@Column.type.serializer)
        } catch (e: DBException) {
            this.status = TransactionStatus.ERROR
            throw TransactionException.TransactionStorageException(this.tid, e.message ?: "Unknown")
        }

        /**
         * Deletes a record from this [Column]. This action will set this [Column.Tx] to [TransactionStatus.DIRTY]
         * and acquire a column-wide write lock until the [Column.Tx] either commit or rollback is issued.
         *
         * @param tupleId The ID of the record that should be deleted
         */
        fun delete(tupleId: Long) = try {
            acquireWriteLock()
            checkValidTupleId(tupleId)
            this@Column.store.delete(tupleId, this@Column.type.serializer)

            /* Update header. */
            val header = this@Column.header
            header.count -= 1
            header.modified = System.currentTimeMillis()
            this@Column.store.update(HEADER_RECORD_ID, header, ColumnHeaderSerializer)
        } catch (e: DBException) {
            this.status = TransactionStatus.ERROR
            throw TransactionException.TransactionStorageException(this.tid, e.message ?: "Unknown")
        }

        /**
         * Deletes all the specified records from this [Column]. This action will set this [Column.Tx] to [TransactionStatus.DIRTY]
         * and acquire a column-wide write lock until the [Column.Tx] either commit or rollback is issued.
         *
         * @param tupleIds The IDs of the records that should be deleted.
         */
        fun deleteAll(tupleIds: Collection<Long>) = try {
            acquireWriteLock()
            tupleIds.forEach{
                checkValidTupleId(it)
                this@Column.store.delete(it, this@Column.type.serializer)
            }

            /* Update header. */
            val header = this@Column.header
            header.count -= tupleIds.size
            header.modified = System.currentTimeMillis()
            store.update(HEADER_RECORD_ID, header, ColumnHeaderSerializer)
        } catch (e: DBException) {
            this.status = TransactionStatus.ERROR
            throw TransactionException.TransactionStorageException(this.tid, e.message ?: "Unknown")
        }

        /**
         * Returns the [ColumnType] of the [Column] associated with this [Column.Tx].
         *
         * @return [ColumnType]
         */
        val type: ColumnType<T>
            get() = this@Column.type

        /**
         * Checks if the provided tupleID is valid. Otherwise, an exception will be thrown.
         */
        private fun checkValidTupleId(tupleId: Long) {
            if ((tupleId < 0L) or (tupleId == HEADER_RECORD_ID)) {
                throw TransactionException.InvalidTupleId(tid, tupleId)
            }
        }

        /**
         * Checks if this [Column.Tx] is still open. Otherwise, an exception will be thrown.
         */
        @Synchronized
        private fun checkValidOrThrow() {
            if (this.status == TransactionStatus.CLOSED) throw TransactionException.TransactionClosedException(tid)
            if (this.status == TransactionStatus.ERROR) throw TransactionException.TransactionInErrorException(tid)
        }

        /**
         * Tries to acquire a write-lock. If method fails, an exception will be thrown
         */
        @Synchronized
        private fun acquireWriteLock() {
            if (this.readonly) throw TransactionException.TransactionReadOnlyException(tid)
            if (this.status == TransactionStatus.CLOSED) throw TransactionException.TransactionClosedException(tid)
            if (this.status == TransactionStatus.ERROR) throw TransactionException.TransactionInErrorException(tid)
            if (this.status != TransactionStatus.DIRTY) {
                if (this@Column.txLock.writeLock().tryLock()) {
                    this.status = TransactionStatus.DIRTY
                } else {
                    throw TransactionException.TransactionWriteLockException(this.tid)
                }
            }
        }
    }

    /**
     * The header data structure of any [Column]
     */
    private class ColumnHeader(val type: ColumnType<*>, var size: Int = 0, var nullable: Boolean = true, var count: Long = 0, var created: Long = System.currentTimeMillis(), var modified: Long = System.currentTimeMillis())

    /**
     * A [Serializer] for [ColumnHeader].
     */
    private object ColumnHeaderSerializer: Serializer<ColumnHeader> {
        override fun serialize(out: DataOutput2, value: ColumnHeader) {
            out.writeUTF(Column.HEADER_IDENTIFIER)
            out.writeShort(Column.HEADER_VERSION.toInt())
            out.writeUTF(value.type.name)
            out.writeInt(value.size)
            out.writeBoolean(value.nullable)
            out.packLong(value.count)
            out.writeLong(value.created)
            out.writeLong(value.modified)
        }

        override fun deserialize(input: DataInput2, available: Int): ColumnHeader {
            if (!this.validate(input)) {
                throw DatabaseException.InvalidFileException("Cottontail DB Column")
            }
            return ColumnHeader(ColumnType.typeForName(input.readUTF()), input.readInt(), input.readBoolean(), input.unpackLong(), input.readLong(), input.readLong())
        }

        /**
         * Validates the [ColumnHeader]. Must be executed before deserialization
         *
         * @return True if validation was successful, false otherwise.
         */
        private fun validate(input: DataInput2): Boolean {
            val identifier = input.readUTF()
            val version = input.readShort()
            return (version == Column.HEADER_VERSION) and (identifier == Column.HEADER_IDENTIFIER)
        }
    }
}
