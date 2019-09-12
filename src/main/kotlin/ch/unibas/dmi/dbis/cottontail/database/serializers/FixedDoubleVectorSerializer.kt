package ch.unibas.dmi.dbis.cottontail.database.serializers

import ch.unibas.dmi.dbis.cottontail.model.values.DoubleVectorValue
import org.mapdb.DataInput2
import org.mapdb.DataOutput2
import org.mapdb.Serializer
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j

/**
 * A [Serializer] for [DoubleVectorValue]s that a fixed in length.
 *
 * @author Ralph Gasser
 * @version 1.0
 */
class FixedDoubleVectorSerializer(val size: Int): Serializer<DoubleVectorValue> {
    override fun serialize(out: DataOutput2, value: DoubleVectorValue) {
        for (i in 0 until size) {
            out.writeDouble(value.value.getDouble(i))
        }
    }
    override fun deserialize(input: DataInput2, available: Int): DoubleVectorValue {
        val vector = Nd4j.createUninitialized(DataType.DOUBLE, this.size.toLong())
        for (i in 0 until size) {
            vector.putScalar(i.toLong(), input.readDouble())
        }
        return DoubleVectorValue(vector)
    }
}