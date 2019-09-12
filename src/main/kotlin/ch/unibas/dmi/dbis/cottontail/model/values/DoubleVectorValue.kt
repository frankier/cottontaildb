package ch.unibas.dmi.dbis.cottontail.model.values

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

inline class DoubleVectorValue(override val value: INDArray) : Value<INDArray> {

    constructor(value: DoubleArray): this(Nd4j.createFromArray(*value))

    override val numeric: Boolean
        get() = false



    override fun compareTo(other: Value<*>): Int {
        throw IllegalArgumentException("DoubleArrayValue can can only be compared for equality.")
    }
}