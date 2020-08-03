package org.vitrivr.cottontail.execution.tasks.entity.projection

import com.github.dexecutor.core.task.Task
import org.vitrivr.cottontail.database.entity.Entity
import org.vitrivr.cottontail.database.general.query
import org.vitrivr.cottontail.execution.tasks.basics.ExecutionTask
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.Name
import org.vitrivr.cottontail.model.recordset.Recordset
import org.vitrivr.cottontail.model.values.BooleanValue

/**
 * A [Task] used during query execution. It takes a single [Entity] and checks if it contains any entries. It thereby creates a 1x1 [Recordset].
 *
 * @author Ralph Gasser
 * @version 1.0.3
 */
class EntityExistsProjectionTask(val entity: Entity, val alias: String? = null) : ExecutionTask("EntityExistsProjectionTask[${entity.name}]") {

    /**
     * Executes this [EntityExistsProjectionTask]
     */
    override fun execute(): Recordset {
        assertNullaryInput()

        val name = if (this.alias != null) {
            Name.ColumnName(this.alias)
        } else {
            this.entity.name.column("exists()")
        }
        val column = arrayOf(ColumnDef.withAttributes(name, "BOOLEAN"))
        return this.entity.Tx(true).query {
            val recordset = Recordset(column, capacity = 1)
            recordset.addRowUnsafe(arrayOf(BooleanValue(it.count() > 0)))
            recordset
        } ?: Recordset(column)
    }
}