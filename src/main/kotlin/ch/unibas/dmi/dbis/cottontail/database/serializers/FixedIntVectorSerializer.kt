package ch.unibas.dmi.dbis.cottontail.database.serializers

import ch.unibas.dmi.dbis.cottontail.model.values.IntVectorValue
import org.mapdb.DataInput2
import org.mapdb.DataOutput2
import org.mapdb.Serializer
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j

/**
 * A [Serializer] for [IntVectorValue]s that are fixed in length.
 *
 * @author Ralph Gasser
 * @version 1.0
 */
class FixedIntVectorSerializer(val size: Int): Serializer<IntVectorValue> {
    override fun serialize(out: DataOutput2, value: IntVectorValue) {
        for (i in 0 until size) {
            out.writeInt(value.value.getInt(i))
        }
    }
    override fun deserialize(input: DataInput2, available: Int): IntVectorValue {
        val vector = Nd4j.createUninitialized(DataType.INT, this.size.toLong())
        for (i in 0 until size) {
            vector.putScalar(i.toLong(), input.readInt())
        }
        return IntVectorValue(vector)
    }
}