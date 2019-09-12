package ch.unibas.dmi.dbis.cottontail.math.knn.metrics

import org.nd4j.linalg.api.ndarray.INDArray
import java.util.*


/**
 * Interface implemented by [DistanceFunction]s that can be used to calculate the distance between two vectors.
 *
 * @author Ralph Gasser
 * @version 1.0
 */
interface DistanceFunction {

    /**
     * Estimation of the number of operations required per vector component.
     */
    val operations: Int;

    /**
     * Calculates the weighted distance between two [FloatArray]s
     *
     * @param a First [FloatArray]
     * @param b Second [FloatArray]
     * @param weights The [FloatArray] containing the weights.
     *
     * @return Distance between a and b.
     */
    operator fun invoke(a: INDArray, b: INDArray, weights: INDArray): Double

    /**
     * Calculates the weighted distance between two [BitSet]s, i.e. [BooleanArray]'s where
     * each element can either be 1 or 0.
     *
     * @param a First [BitSet]
     * @param b Second [BitSet]
     * @param weights The [Float] containing the weights.
     *
     * @return Distance between a and b.
     */
    operator fun invoke(a: BitSet, b: BitSet, weights: FloatArray): Double

    /**
     * Calculates the distance between two [FloatArray]s
     *
     * @param a First [FloatArray]
     * @param b Second [FloatArray]
     * @return Distance between a and b.
     */
    operator fun invoke(a: INDArray, b: INDArray): Double

    /**
     * Calculates the distance between two [BitSet]s, i.e. [BooleanArray]'s where
     * each element can either be 1 or 0.
     *
     * @param a First [BitSet]
     * @param b Second [BitSet]
     *
     * @return Distance between a and b.
     */
    operator fun invoke(a: BitSet, b: BitSet): Double
}