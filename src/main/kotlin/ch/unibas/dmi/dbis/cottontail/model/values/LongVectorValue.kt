package ch.unibas.dmi.dbis.cottontail.model.values

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

inline class LongVectorValue(override val value: INDArray) : Value<INDArray> {
    constructor(value: LongArray): this(Nd4j.createFromArray(*value))

    override val numeric: Boolean
        get() = false

    override fun compareTo(other: Value<*>): Int {
        throw IllegalArgumentException("FloatArrayValue can can only be compared for equality.")
    }
}