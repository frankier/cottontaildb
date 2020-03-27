package ch.unibas.dmi.dbis.cottontail.database.serializers

import ch.unibas.dmi.dbis.cottontail.model.values.DoubleVectorValue
import ch.unibas.dmi.dbis.cottontail.storage.engine.hare.basics.Page
import ch.unibas.dmi.dbis.cottontail.storage.engine.hare.serializer.Serializer
import org.mapdb.DataInput2
import org.mapdb.DataOutput2

/**
 * A [Serializer] for [DoubleVectorValue]s that a fixed in length.
 *
 * @author Ralph Gasser
 * @version 1.0
 */
class FixedDoubleVectorSerializer(override val logicalSize: Int): Serializer<DoubleVectorValue> {

    override val physicalSize: Int = Long.SIZE_BYTES * this.logicalSize

    override fun serialize(out: DataOutput2, value: DoubleVectorValue) {
        for (i in 0 until this.logicalSize) {
            out.writeDouble(value[i].value)
        }
    }
    override fun deserialize(input: DataInput2, available: Int): DoubleVectorValue {
        val vector = DoubleArray(this.logicalSize)
        for (i in 0 until this.logicalSize) {
            vector[i] = input.readDouble()
        }
        return DoubleVectorValue(vector)
    }

    override fun serialize(page: Page, offset: Int, value: DoubleVectorValue) {
        page.putBytes(offset, value.data)
    }

    override fun deserialize(page: Page, offset: Int): DoubleVectorValue {
        return DoubleVectorValue(page.getSlice(offset, offset + this.physicalSize))
    }
}