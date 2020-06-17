package org.vitrivr.cottontail.math.knn.metrics

import org.vitrivr.cottontail.model.values.DoubleValue
import org.vitrivr.cottontail.model.values.types.VectorValue
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * The Haversine Distance, based on the Haversine Formula, which determines the great-circle distance between two points on a sphere.
 * Se the wiki entry: https://en.wikipedia.org/wiki/Haversine_formula
 * <br>
 * This is only applicable to two-dimensional vectors with earth coordinates in degrees.
 *
 * @version 1.0
 * @author Loris Sauter
 */
object HaversineDistance : MinkowskiDistance {
    override val p = 2
    override val cost = 1f

    private const val EARTH_RADIUS_APPROX = 6371e3

    private fun haversine(aLat: Double, aLon: Double, bLat: Double, bLon: Double):Double{
        val phi1 = StrictMath.toRadians(aLat)
        val phi2 = StrictMath.toRadians(bLat)
        val deltaPhi = StrictMath.toRadians(bLat - aLat)
        val deltaLambda = StrictMath.toRadians(bLon - aLon)
        val c = sin(deltaPhi/2.0) * sin(deltaPhi /2.0) + cos(phi1) * cos(phi2) * sin(deltaLambda / 2.0) * sin(deltaLambda / 2.0)
        val d = 2.0 * atan2(sqrt(c), sqrt(1-c))
        return EARTH_RADIUS_APPROX * d
    }

    override fun invoke(a: VectorValue<*>, b: VectorValue<*>): DoubleValue = DoubleValue(haversine(a[0].asDouble().value, a[1].asDouble().value, b[0].asDouble().value, b[1].asDouble().value))

    override fun invoke(a: VectorValue<*>, b: VectorValue<*>, weights: VectorValue<*>): DoubleValue = this.invoke(a, b)
}