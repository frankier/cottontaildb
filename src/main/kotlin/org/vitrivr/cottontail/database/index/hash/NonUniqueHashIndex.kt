package org.vitrivr.cottontail.database.index.hash

import org.mapdb.DBMaker
import org.mapdb.HTreeMap
import org.mapdb.Serializer
import org.vitrivr.cottontail.database.column.Column
import org.vitrivr.cottontail.database.entity.Entity
import org.vitrivr.cottontail.database.events.DataChangeEvent
import org.vitrivr.cottontail.database.events.DataChangeEventType
import org.vitrivr.cottontail.database.index.Index
import org.vitrivr.cottontail.database.index.IndexType
import org.vitrivr.cottontail.database.queries.components.AtomicBooleanPredicate
import org.vitrivr.cottontail.database.queries.components.ComparisonOperator
import org.vitrivr.cottontail.database.queries.components.Predicate
import org.vitrivr.cottontail.database.queries.planning.cost.Cost
import org.vitrivr.cottontail.database.queries.planning.cost.Costs
import org.vitrivr.cottontail.database.schema.Schema
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.Name
import org.vitrivr.cottontail.model.basics.Record
import org.vitrivr.cottontail.model.exceptions.QueryException
import org.vitrivr.cottontail.model.exceptions.ValidationException
import org.vitrivr.cottontail.model.recordset.Recordset
import org.vitrivr.cottontail.model.values.types.Value
import org.vitrivr.cottontail.utilities.extensions.write
import java.nio.file.Path

/**
 * Represents an index in the Cottontail DB data model. An [Index] belongs to an [Entity] and can be used to index one to many
 * [Column]s. Usually, [Index]es allow for faster data access. They process [Predicate]s and return [Recordset]s.
 *
 * @see Schema
 * @see Column
 * @see Entity.Tx
 *
 * @author Luca Rossetto
 * @version 1.1
 */
class NonUniqueHashIndex(override val name: Name.IndexName, override val parent: Entity, override val columns: Array<ColumnDef<*>>) : Index() {
    /**
     * Index-wide constants.
     */
    companion object {
        const val MAP_FIELD_NAME = "nu_map"
    }

    /** Path to the [NonUniqueHashIndex] file. */
    override val path: Path = this.parent.path.resolve("idx_nu_$name.db")

    /** The type of [Index] */
    override val type: IndexType = IndexType.HASH_UQ

    /** The [NonUniqueHashIndex] implementation returns exactly the columns that is indexed. */
    override val produces: Array<ColumnDef<*>> = this.columns

    /** The internal [DB] reference. */
    private val db = if (parent.parent.parent.config.memoryConfig.forceUnmapMappedFiles) {
        DBMaker.fileDB(this.path.toFile()).fileMmapEnable().cleanerHackEnable().transactionEnable().make()
    } else {
        DBMaker.fileDB(this.path.toFile()).fileMmapEnable().transactionEnable().make()
    }

    /** Map structure used for [NonUniqueHashIndex]. */
    private val map: HTreeMap<out Value, LongArray> = this.db.hashMap(MAP_FIELD_NAME, this.columns.first().type.serializer(this.columns.size), Serializer.LONG_ARRAY).counterEnable().createOrOpen()

    /**
     * Flag indicating if this [NonUniqueHashIndex] has been closed.
     */
    @Volatile
    override var closed: Boolean = false
        private set

    /**
     * Performs a lookup through this [NonUniqueHashIndex].
     *
     * @param predicate The [Predicate] for the lookup
     * @param tx Reference to the [Entity.Tx] the call to this method belongs to.
     *
     * @return The resulting [Recordset]
     */
    override fun filter(predicate: Predicate, tx: Entity.Tx): Recordset = if (predicate is AtomicBooleanPredicate<*>) {
        /* Create empty recordset. */
        val recordset = Recordset(this.columns)

        /* Fetch the columns that match the predicate. */
        val results = when (predicate.operator) {
            ComparisonOperator.IN -> predicate.values.map { it to this.map[it] }.toMap()
            ComparisonOperator.EQUAL -> mapOf(Pair(predicate.values.first(), this.map[predicate.values.first()]))
            else -> throw QueryException.IndexLookupFailedException(this.name, "Instance of unique hash-index does not support ${predicate.operator} comparison operators.")
        }

        /* Generate record set .*/
        if (predicate.not) {
            this.map.forEach { (value, tid) ->
                if (results.containsKey(value)) {
                    tid.forEach {
                        recordset.addRowUnsafe(tupleId = it, values = arrayOf(value))
                    }
                }
            }
        } else {
            results.forEach {
                it.value?.forEach { id ->
                    recordset.addRowUnsafe(tupleId = id, values = arrayOf(it.key))
                }
            }
        }
        recordset
    } else {
        throw QueryException.UnsupportedPredicateException("Index '${this.name}' (non-unique hash-index) does not support predicates of type '${predicate::class.simpleName}'.")
    }

    /**
     * Checks if the provided [Predicate] can be processed by this instance of [NonUniqueHashIndex]. [NonUniqueHashIndex] can be used to process IN and EQUALS
     * comparison operations on the specified column
     *
     * @param predicate The [Predicate] to check.
     * @return True if [Predicate] can be processed, false otherwise.
     */
    override fun canProcess(predicate: Predicate): Boolean = if (predicate is AtomicBooleanPredicate<*>) {
        predicate.columns.first() == this.columns[0] && (predicate.operator == ComparisonOperator.IN || predicate.operator == ComparisonOperator.EQUAL)
    } else {
        false
    }

    /**
     * Calculates the cost estimate of this [NonUniqueHashIndex] processing the provided [Predicate].
     *
     * @param predicate [Predicate] to check.
     * @return Cost estimate for the [Predicate]
     */
    override fun cost(predicate: Predicate): Cost = when {
        predicate !is AtomicBooleanPredicate<*> || predicate.columns.first() != this.columns[0] -> Cost.INVALID
        predicate.operator == ComparisonOperator.EQUAL -> Cost(Costs.DISK_ACCESS_READ, Costs.MEMORY_ACCESS_READ, predicate.columns.map { it.physicalSize }.sum().toFloat())
        predicate.operator == ComparisonOperator.IN -> Cost(Costs.DISK_ACCESS_READ, Costs.MEMORY_ACCESS_READ, predicate.columns.map { it.physicalSize }.sum().toFloat()) * predicate.values.size
        else -> Cost.INVALID
    }

    /**
     * Returns true since [NonUniqueHashIndex] supports incremental updates.
     *
     * @return True
     */
    override fun supportsIncrementalUpdate(): Boolean = true

    /**
     * (Re-)builds the [NonUniqueHashIndex].
     *
     * @param tx Reference to the [Entity.Tx] the call to this method belongs to.
     */
    override fun rebuild(tx: Entity.Tx) {
        /* Clear existing map. */
        this.map.clear()

        /* (Re-)create index entries. */
        val localMap = mutableMapOf<Value, MutableList<Long>>()
        tx.forEach {
            val value = it[this.columns[0]]
                    ?: throw ValidationException.IndexUpdateException(this.name, "A value cannot be null for instances of non-unique hash-index but tid=${it.tupleId} is")
            if (!localMap.containsKey(value)) {
                localMap[value] = mutableListOf(it.tupleId)
            } else {
                localMap[value]!!.add(it.tupleId)
            }
        }
        val castMap = this.map as HTreeMap<Value,LongArray>
        localMap.forEach { (value, l) -> castMap[value] = l.toLongArray() }
        this.db.commit()
    }

    /**
     * Updates the [NonUniqueHashIndex] with the provided [Record]. This method determines, whether the [Record] should be added or updated
     *
     * @param record Record to update the [NonUniqueHashIndex] with.
     * @param tx Reference to the [Entity.Tx] the call to this method belongs to.
     */
    override fun update(update: Collection<DataChangeEvent>, tx: Entity.Tx) = try {
        val localMap = this.map as HTreeMap<Value,LongArray>

        /* Define action for inserting an entry based on a DataChangeEvent. */
        val atomicInsert = { event: DataChangeEvent ->
            val newValue = event.new?.get(this.columns[0])
                    ?: throw ValidationException.IndexUpdateException(this.name, "Values cannot be null for instances of UniqueHashIndex but tid=${event.new?.tupleId} is.")
            if (localMap.containsKey(newValue)) {
                val oldArray = localMap[newValue]!!
                if (!oldArray.contains(event.new.tupleId)) {
                    val newArray = oldArray.copyOf(oldArray.size + 1)
                    newArray[oldArray.size] = event.new.tupleId
                    localMap[newValue] = newArray
                }
            } else {
                localMap[newValue] = longArrayOf(event.new.tupleId)
            }
        }

        /* Define action for deleting an entry based on a DataChangeEvent. */
        val atomicDelete = { event: DataChangeEvent ->
            val oldValue = event.old?.get(this.columns[0])
                    ?: throw ValidationException.IndexUpdateException(this.name, "Values cannot be null for instances of UniqueHashIndex but tid=${event.new?.tupleId} is.")
            if (localMap.containsKey(oldValue)) {
                val oldArray = localMap[oldValue]!!
                if (oldArray.contains(event.old.tupleId)) {
                    localMap[oldValue] = oldArray.filter { it != event.old.tupleId }.toLongArray()
                }
            }
        }

        /* Process the DataChangeEvents. */
        loop@ for (event in update) {
            when (event.type) {
                DataChangeEventType.INSERT -> atomicInsert(event)
                DataChangeEventType.UPDATE -> {
                    if (event.new?.get(this.columns[0]) != event.old?.get(this.columns[0])) {
                        atomicDelete(event)
                        atomicInsert(event)
                    }
                }
                DataChangeEventType.DELETE -> atomicDelete(event)
                else -> continue@loop
            }
        }

        /* Commit the change. */
        this.db.commit()
    } catch (e: Throwable) {
        this.db.rollback()
        throw e
    }

    /**
     * Closes this [NonUniqueHashIndex] and the associated data structures.
     */
    override fun close() = this.globalLock.write {
        if (!this.closed) {
            this.db.close()
            this.closed = true
        }
    }
}
