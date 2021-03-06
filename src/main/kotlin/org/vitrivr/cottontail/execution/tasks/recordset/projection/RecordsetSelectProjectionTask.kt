package org.vitrivr.cottontail.execution.tasks.recordset.projection

import com.github.dexecutor.core.task.Task
import com.github.dexecutor.core.task.TaskExecutionException
import org.vitrivr.cottontail.database.entity.Entity
import org.vitrivr.cottontail.execution.tasks.basics.ExecutionTask
import org.vitrivr.cottontail.model.recordset.Recordset
import org.vitrivr.cottontail.utilities.name.Match
import org.vitrivr.cottontail.utilities.name.Name

/**
 * A [Task] used during query execution. It takes a single [Recordset] as input and fetches the specified [Column][org.vitrivr.cottontail.database.column.Column]
 * values from the specified [Entity] in order to addRow these values to the resulting [Recordset].
 *
 * A [Recordset] that undergoes this task will grow in terms of columns it contains. However, it should not affect the number of rows.
 *
 * @author Ralph Gasser
 * @version 1.1
 */
class RecordsetSelectProjectionTask(val fields: Map<Name, Name?>) : ExecutionTask("RecordsetSelectProjectionTask") {
    /**
     * Executes this [RecordsetSelectProjectionTask]
     */
    override fun execute(): Recordset {
        assertUnaryInput()

        /* Get records from parent task. */
        val parent = this.first()
                ?: throw TaskExecutionException("SELECT projection could not be executed because parent task has failed.")

        /* Find longest, common prefix of all projection sub-clauses. */
        val longestCommonPrefix = Name.findLongestCommonPrefix(this.fields.keys)

        /* Determine columns to rename. */
        val dropIndex = mutableListOf<Int>()
        val renameIndex = mutableListOf<Pair<Int, Name>>()
        parent.columns.forEachIndexed { i, c ->
            var match = false
            this.fields.forEach { (k, v) ->
                val m = k.match(c.name)
                if (m != Match.NO_MATCH) {
                    match = true
                    if (v != null) {
                        renameIndex.add(Pair(i, v))
                    } else if (longestCommonPrefix != Name.COTTONTAIL_ROOT_NAME) {
                        renameIndex.add(Pair(i, c.name.normalize(longestCommonPrefix)))
                    }
                }
            }
            if (!match) dropIndex.add(i)
        }

        /* Create new Recordset with specified columns and return it . */
        return if (dropIndex.size > 0 && renameIndex.size > 0) {
            parent.renameColumnsWithIndex(renameIndex).dropColumnsWithIndex(dropIndex)
        } else if (renameIndex.size > 0) {
            parent.renameColumnsWithIndex(renameIndex)
        } else if (dropIndex.size > 0) {
            parent.dropColumnsWithIndex(dropIndex)
        } else {
            return parent
        }
    }
}