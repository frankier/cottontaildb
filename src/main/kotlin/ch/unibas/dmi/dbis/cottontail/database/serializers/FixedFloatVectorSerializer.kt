package ch.unibas.dmi.dbis.cottontail.database.serializers

import ch.unibas.dmi.dbis.cottontail.model.values.FloatVectorValue
import org.mapdb.DataInput2
import org.mapdb.DataOutput2
import org.mapdb.Serializer
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j

/**
 * A [Serializer] for [FloatVectorValue]s that a fixed in length.
 *
 * @author Ralph Gasser
 * @version 1.0
 */
class FixedFloatVectorSerializer(val size: Int): Serializer<FloatVectorValue> {
    override fun serialize(out: DataOutput2, value: FloatVectorValue) {
        for (i in 0 until size) {
            out.writeFloat(value.value.getFloat(IntArray(1) {i}))
        }
    }
    override fun deserialize(input: DataInput2, available: Int): FloatVectorValue {
        val vector = Nd4j.createUninitialized(DataType.FLOAT, this.size.toLong())
        for (i in 0 until size) {
            vector.putScalar(i.toLong(), input.readFloat())
        }
        return FloatVectorValue(vector)
    }
}