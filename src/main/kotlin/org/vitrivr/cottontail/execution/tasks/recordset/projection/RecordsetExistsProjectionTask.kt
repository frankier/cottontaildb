package org.vitrivr.cottontail.execution.tasks.recordset.projection

import com.github.dexecutor.core.task.Task
import com.github.dexecutor.core.task.TaskExecutionException
import org.vitrivr.cottontail.execution.tasks.basics.ExecutionTask
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.basics.Name
import org.vitrivr.cottontail.model.recordset.Recordset
import org.vitrivr.cottontail.model.values.BooleanValue

/**
 * A [Task] used during query execution. It takes a single [Recordset] as input, counts the number of of rows and returns it as [Recordset].
 *
 * @author Ralph Gasser
 * @version 1.0.1
 */
class RecordsetExistsProjectionTask : ExecutionTask("RecordsetExistsProjectionTask") {
    /**
     * Executes this [RecordsetExistsProjectionTask]
     */
    override fun execute(): Recordset {
        assertUnaryInput()

        /* Get records from parent task. */
        val parent = this.first()
                ?: throw TaskExecutionException("EXISTS projection could not be executed because parent task has failed.")

        /* Create new Recordset with new columns. */
        val recordset = Recordset(arrayOf(ColumnDef.withAttributes(parent.columns.first().name.entity()?.column("exists(*)")
                ?: Name.ColumnName("exists()"), "BOOLEAN")))
        recordset.addRowUnsafe(arrayOf(BooleanValue(parent.rowCount > 0)))
        return recordset
    }
}