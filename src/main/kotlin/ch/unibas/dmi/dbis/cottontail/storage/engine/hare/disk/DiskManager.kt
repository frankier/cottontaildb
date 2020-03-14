package ch.unibas.dmi.dbis.cottontail.storage.engine.hare.disk

import ch.unibas.dmi.dbis.cottontail.storage.basics.MemorySize
import ch.unibas.dmi.dbis.cottontail.storage.basics.Units
import ch.unibas.dmi.dbis.cottontail.storage.engine.hare.DataCorruptionException
import ch.unibas.dmi.dbis.cottontail.storage.engine.hare.FileLockException
import ch.unibas.dmi.dbis.cottontail.storage.engine.hare.PageIdOutOfBoundException
import ch.unibas.dmi.dbis.cottontail.storage.engine.hare.disk.Constants.FILE_HEADER_IDENTIFIER
import ch.unibas.dmi.dbis.cottontail.storage.engine.hare.disk.Constants.FILE_HEADER_VERSION
import ch.unibas.dmi.dbis.cottontail.storage.engine.hare.disk.Constants.FILE_SANITY_CHECK
import ch.unibas.dmi.dbis.cottontail.storage.engine.hare.disk.Constants.FILE_SANITY_OK
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.channels.FileLock
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.util.concurrent.atomic.AtomicBoolean
import java.util.zip.CRC32C

/**
 * The [DiskManager] facilitates reading and writing of [Page]s from/to the underlying disk storage. Only one
 * [DiskManager] can be opened per HARE file and it acquires an exclusive [FileLock] once created.
 *
 * The [DiskManager] only transfers bytes from and to [Page]s. The management of [Page]s is handled by the [BufferPool].
 *
 * @see BufferPool
 *
 * @version 1.0
 * @author Ralph Gasser
 */
abstract class DiskManager(val path: Path, val lockTimeout: Long) : AutoCloseable {
    /** The [FileChannel] used to access the file managed by this [DiskManager]. */
    protected val fileChannel: FileChannel = FileChannel.open(this.path, StandardOpenOption.READ, StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.DSYNC, StandardOpenOption.SPARSE)

    /** Acquires an exclusive [FileLock] for file underlying this [FileChannel]. Makes sure, that no other process uses the same HARE file. */
    protected val fileLock = FileUtilities.acquireFileLock(this.fileChannel, this.lockTimeout)

    /** Flag indicating, whether this [DiskManager] was closed. */
    protected val closed: AtomicBoolean = AtomicBoolean(false)

    /** Accessor to the [Header] of the HARE file managed by this [DiskManager]. */
    protected val header = Header(this.fileChannel.size() == 0L)

    /** Returns the size of the HARE file managed by this [DiskManager]. */
    val size
        get() = MemorySize(this.fileChannel.size().toDouble(), Units.BYTE)

    /** Number of [Page]s held by the HARE file managed by this [DiskManager]. */
    val pages
        get() = this.header.pages

    /**
     * Fetches the data identified by the given [PageId] into the given [Page] object thereby replacing the content of that [Page].
     *
     * @param id [PageId] to fetch data for.
     * @param page [Page] to fetch data into. Its content will be updated.
     */
    abstract fun read(id: PageId, page: Page)

    /**
     * Updates the [Page] in the HARE file managed by this [DirectDiskManager].
     *
     * @param page [Page] to update.
     */
    abstract fun update(page: Page)

    /**
     * Allocates new [Page] in the HARE file managed by this [DirectDiskManager].
     *
     * @param page [Page] to append. Its [PageId] and flags will be updated.
     */
    abstract fun allocate(page: Page)

    /**
     * Frees the [Page] identified by the given [PageId].
     *
     * @param pageId The [PageId] of the [Page] that should be freed.
     */
    abstract fun free(pageId: PageId)

    /**
     * Commits all changes made through this [DiskManager].
     */
    abstract fun commit()

    /**
     * Rolls back all changes made through this [DiskManager].
     */
    abstract fun rollback()

    /**
     * Closes this [DiskManager], releasing the underlying file.
     */
    override fun close() {
        if (this.closed.compareAndSet(false, true)) {
            this.fileLock.release()
            this.fileChannel.close()
        }
    }

    /**
     * Converts the given [PageId] to an offset into the file managed by this [DirectDiskManager]. Calling this method
     * also makes necessary sanity checks regarding the file's channel status and pageId bounds.
     *
     * @param pageId The [PageId] to translate to a position.
     * @return The offset into the file.
     */
    protected fun pageIdToPosition(pageId: PageId): Long {
        if (this.closed.get()) throw IllegalStateException("DiskManager for {${this.path}} was closed and cannot be used to access data.")
        if (pageId > this.header.pages || pageId < 1) throw PageIdOutOfBoundException(pageId, this)
        return pageId shl Page.Constants.PAGE_BIT_SHIFT
    }

    /**
     * The [Header] or 0-Page of this HARE file.
     *
     * @version 1.0
     * @author Ralph Gasser
     */
    protected inner class Header(new: Boolean) {
        /** A fixed 4096 byte [ByteBuffer] used to provide access to the header of this HARE file managed by this [DiskManager]. */
        val buffer: ByteBuffer = ByteBuffer.allocateDirect(Page.Constants.PAGE_DATA_SIZE_BYTES)

        init {
            if (new) {
                this.buffer.putChar(FILE_HEADER_IDENTIFIER[0])             /* 0: Identifier H. */
                this.buffer.putChar(FILE_HEADER_IDENTIFIER[1])             /* 2: Identifier A. */
                this.buffer.putChar(FILE_HEADER_IDENTIFIER[2])             /* 4: Identifier R. */
                this.buffer.putChar(FILE_HEADER_IDENTIFIER[3])             /* 6: Identifier E. */
                this.buffer.putInt(FileType.DEFAULT.ordinal)               /* 8: Type of HARE file. */
                this.buffer.put(FILE_HEADER_VERSION)                       /* 12: Version of the HARE format. */
                this.buffer.put(FILE_SANITY_OK)                            /* 13: Sanity byte; 0 if file was properly closed, 1 if not.  */
                this.buffer.putLong(0L)                              /* 14: Page counter; number of pages. */
                this.buffer.putInt(0)                                /* 22: Page counter; number of freed pages. */
                this.buffer.putLong(0L)                              /* 26: CRC32 checksum for HARE file. */
                this.buffer.putLong(-1L)                              /* 26: Time of last WAL (ignored for DirectDiskManager). */
            } else {
                /* Read the file header. */
                this@DiskManager.fileChannel.read(this.buffer, 0L)
                this.buffer.rewind()

                /** Make necessary check on startup. */
                require(this.buffer.char == FILE_HEADER_IDENTIFIER[0]) { DataCorruptionException("HARE identifier missing in HARE file ${this@DiskManager.path.fileName}.") }
                require(this.buffer.char == FILE_HEADER_IDENTIFIER[1]) { DataCorruptionException("HARE identifier missing in HARE file ${this@DiskManager.path.fileName}.") }
                require(this.buffer.char == FILE_HEADER_IDENTIFIER[2]) { DataCorruptionException("HARE identifier missing in HARE file ${this@DiskManager.path.fileName}.") }
                require(this.buffer.char == FILE_HEADER_IDENTIFIER[3]) { DataCorruptionException("HARE identifier missing in HARE file ${this@DiskManager.path.fileName}.") }
                require(this.buffer.int == FileType.DEFAULT.ordinal)
                require(this.buffer.get() == FILE_HEADER_VERSION) { DataCorruptionException("HARE file version is incorrect in HARE file ${this@DiskManager.path.fileName}.") }
                require(this.pages >= 0) { DataCorruptionException("Negative number of allocated pages found in HARE file ${this@DiskManager.path.fileName}.") }
                require(this.freed >= 0) { DataCorruptionException("Negative number of freed pages found in HARE file ${this@DiskManager.path.fileName}.") }

                if (this.buffer.get() != FILE_SANITY_OK) {
                    val page = Page(ByteBuffer.allocateDirect(Page.Constants.PAGE_DATA_SIZE_BYTES))
                    val crc32 = CRC32C()
                    for (i in 1..this.pages) {
                        this@DiskManager.read(i, page)
                        crc32.update(page.data)
                        require(crc32.value == this.checksum) { DataCorruptionException("CRC32C checksum not correct (expected: ${this.checksum}, found: ${crc32.value}) of HARE file ${this@DiskManager.path.fileName}.") }
                    }
                }

                /* Updates sanity flag. */
                this.buffer.put(13, FILE_SANITY_CHECK)
            }

            /* Flush header to disk. */
            this.flush()
        }

        /** Total number of [Page]s managed by this [DiskManager]. */
        var pages: Long
            get() = this.buffer.getLong(14)
            set(v) {
                this.buffer.putLong(14, v)
            }

        /** Total number of freed [Page]s managed by this [DiskManager]. */
        var freed: Int
            get() = this.buffer.getInt(22)
            set(v) {
                this.buffer.putInt(22, v)
            }

        /** CRC32C checksum for the HARE file. */
        var checksum: Long
            get() = this.buffer.getLong(26)
            set(v) {
                this.buffer.putLong(26, v)
            }

        /** Timestamp of last WAL. Only used for [WALDiskManager]. */
        var walTimestamp: Long
            get() = this.buffer.getLong(34)
            set(v) {
                this.buffer.putLong(34, v)
            }

        /** Total number of used [Page]s. */
        val used: Long
            get() {
                return this.buffer.getLong(10) - this. buffer.getInt(18)
            }

        /**
         * Flushes the content of this [Header] to disk.
         */
        fun flush() {
            this@DiskManager.fileChannel.write(this.buffer.rewind(), 0)
            this.buffer.rewind()
        }
    }
}