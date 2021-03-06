package org.vitrivr.cottontail.math.knn.metrics

import org.vitrivr.cottontail.model.values.DoubleValue
import org.vitrivr.cottontail.model.values.types.VectorValue

/**
 * Calculates the Hamming distance between to vectors.
 *
 * @version 1.0
 * @author Ralph Gasser
 */
object HammingDistance : DistanceKernel {
    override val cost = 1.0f


    /**
     * Calculates the Hamming distance between two [VectorValue]s.
     *
     * @param a First [VectorValue]
     * @param b Second [VectorValue]
     *
     * @return Distance between a and b.
     */
    override operator fun invoke(a: VectorValue<*>, b: VectorValue<*>): DoubleValue {
        TODO()
    }

    /**
     * Calculates the weighted Hamming distance between two [VectorValue]s.
     *
     * @param a First [VectorValue]
     * @param b Second [VectorValue]
     * @param weights A list of weights
     *
     * @return Distance between a and b.
     */
    override operator fun invoke(a: VectorValue<*>, b: VectorValue<*>, weights: VectorValue<*>): DoubleValue {
        TODO()
    }
}