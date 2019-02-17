package ch.unibas.dmi.dbis.cottontail.execution.tasks.entity.projection

import ch.unibas.dmi.dbis.cottontail.database.entity.Entity
import ch.unibas.dmi.dbis.cottontail.database.general.query
import ch.unibas.dmi.dbis.cottontail.execution.tasks.ExecutionTask
import ch.unibas.dmi.dbis.cottontail.model.basics.ColumnDef
import ch.unibas.dmi.dbis.cottontail.model.basics.Recordset
import com.github.dexecutor.core.task.Task

/**
 * A [Task] used during query execution. It takes a single [Entity] as input, counts the number of of rows and returns it as [Recordset].
 *
 * @author Ralph Gasser
 * @version 1.0
 */
internal class EntityCountProjectionTask (val entity: Entity, val alias: String? = null): ExecutionTask("EntityCountProjectionTask[${entity.fqn}") {
    override fun execute(): Recordset {
        assertNullaryInput()

        val column = arrayOf(ColumnDef.withAttributes(alias ?: "count(*)", "INTEGER"))
        return this.entity.Tx(true).query {
            val recordset = Recordset(column)
            recordset.addRow(it.count())
            recordset
        } ?: Recordset(column)
    }
}