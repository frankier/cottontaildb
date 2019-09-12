package ch.unibas.dmi.dbis.cottontail.database.serializers

import ch.unibas.dmi.dbis.cottontail.model.values.LongVectorValue
import org.mapdb.DataInput2
import org.mapdb.DataOutput2
import org.mapdb.Serializer
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j

/**
 * A [Serializer] for [LongVectorValue]s that are fixed in length.
 *
 * @author Ralph Gasser
 * @version 1.0
 */
class FixedLongVectorSerializer(val size: Int): Serializer<LongVectorValue> {
    override fun serialize(out: DataOutput2, value: LongVectorValue) {
        for (i in 0 until size) {
            out.writeLong(value.value.getLong(i.toLong()))
        }
    }
    override fun deserialize(input: DataInput2, available: Int): LongVectorValue {
        val vector = Nd4j.createUninitialized(DataType.LONG, this.size.toLong())
        for (i in 0 until size) {
            vector.putScalar(i.toLong(), input.readLong().toInt())
        }
        return LongVectorValue(vector)
    }
}