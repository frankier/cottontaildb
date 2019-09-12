package ch.unibas.dmi.dbis.cottontail.model.values

import org.nd4j.linalg.api.ndarray.INDArray
import java.util.*


/**
 * This is an abstraction over the existing primitive types provided by Kotlin. It allows for the advanced  type system implemented by Cottontail DB.
 *
 * @author Ralph Gasser
 * @version 1.0
 */
interface Value<T> {

    /** Actual value of this [Value]. */
    val value: T

    /** Size of this [Value]. -1 for all types except for vector types. */
    val size
        get() = when (this.value) {
            is INDArray -> (value as INDArray).shape()[0].toInt()
            is BitSet -> (value as BitSet).length()
            else -> -1
        }

    /** True if this is a numeric [Value]. */
    val numeric: Boolean

    /**
     * Comparison operator between two values. Returns -1, 0 or 1 of other value is smaller, equal or greater than this value.
     *
     * @param other Value to compare to.
     * @return -1, 0 or 1 of other value is smaller, equal or greater than this value
     */
    operator fun compareTo(other: Value<*>): Int
}