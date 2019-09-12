package ch.unibas.dmi.dbis.cottontail.execution.tasks.entity.knn

import ch.unibas.dmi.dbis.cottontail.database.column.ColumnType
import ch.unibas.dmi.dbis.cottontail.database.entity.Entity
import ch.unibas.dmi.dbis.cottontail.database.general.begin
import ch.unibas.dmi.dbis.cottontail.database.queries.BooleanPredicate
import ch.unibas.dmi.dbis.cottontail.database.queries.KnnPredicate
import ch.unibas.dmi.dbis.cottontail.execution.tasks.basics.ExecutionTask
import ch.unibas.dmi.dbis.cottontail.math.knn.ComparablePair
import ch.unibas.dmi.dbis.cottontail.math.knn.HeapSelect
import ch.unibas.dmi.dbis.cottontail.model.basics.ColumnDef
import ch.unibas.dmi.dbis.cottontail.model.recordset.Recordset
import ch.unibas.dmi.dbis.cottontail.model.values.DoubleValue
import com.github.dexecutor.core.task.Task
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance
import org.nd4j.linalg.api.ops.impl.reduce3.ManhattanDistance
import org.nd4j.linalg.factory.Nd4j

/**
 * A [Task] that executes a parallel boolean kNN on a double [Column] of the specified [Entity].
 * Parallelism is achieved through the use of co-routines.
 *
 * @author Ralph Gasser
 * @version 1.0
 */
internal class ParallelEntityScanKnnTask(val entity: Entity, val knn: KnnPredicate<INDArray>, val predicate: BooleanPredicate? = null, val parallelism: Short = 2) : ExecutionTask("ParallelEntityScanKnnTask[${entity.fqn}][${knn.column.name}][${knn.distance::class.simpleName}][${knn.k}][q=${knn.query.hashCode()}]") {

    /** Set containing the kNN values. */
    private val knnSet = knn.query.map { HeapSelect<ComparablePair<Long,Double>>(this.knn.k) }

    /** List of the [ColumnDef] this instance of [ParallelEntityScanKnnTask] produces. */
    private val produces: Array<ColumnDef<*>> = arrayOf(ColumnDef("${entity.fqn}.distance", ColumnType.forName("DOUBLE")))

    /** The cost of this [ParallelEntityScanKnnTask] is constant */
    override val cost = entity.statistics.columns * (knn.operations + (predicate?.operations ?: 0)).toFloat() / parallelism
    
    /**
     * Executes this [ParallelEntityScanKnnTask]
     */
    override fun execute(): Recordset {
        /* Extract the necessary data. */
        val columns = arrayOf<ColumnDef<*>>(this.knn.column).plus(predicate?.columns?.toTypedArray() ?: emptyArray())

        /* Make some calculations (number of columns, numberOfQueries, size of a batch). */
        val stride = this.entity.statistics.maxTupleId / this.parallelism
        val numberOfQueries = this@ParallelEntityScanKnnTask.knn.query.size

        /*  Prepare INDArray for query. */
        val exec = Nd4j.getExecutioner()

        /* Execute kNN lookup. */
        this.entity.Tx(readonly = true, columns = columns).begin { tx ->
            runBlocking {
                val jobs = Array(parallelism.toInt()) { j ->
                    GlobalScope.launch {
                        val start = stride * j + 1
                        val end = stride * j + 1 + stride - 1
                        tx.forEach(start, end) { it ->
                            if (this@ParallelEntityScanKnnTask.predicate == null || this@ParallelEntityScanKnnTask.predicate.matches(it)) {
                                val v = it[this@ParallelEntityScanKnnTask.knn.column]
                                if (v != null) {
                                    for (i in 0 until numberOfQueries) {
                                        val op = ManhattanDistance(this@ParallelEntityScanKnnTask.knn.query[i], v.value)
                                        this@ParallelEntityScanKnnTask.knnSet[i].add(ComparablePair(it.tupleId, exec.execAndReturn(op).finalResult.toDouble()))
                                    }
                                }
                            }
                        }
                    }
                }
                jobs.forEach { it.join() }
            }
            true
        }

        /* Generate dataset and return it. */
        val dataset = Recordset(this.produces)
        for (knn in this.knnSet) {
            for (i in 0 until knn.size) {
                dataset.addRowUnsafe(knn[i].first, arrayOf(DoubleValue(knn[i].second)))
            }
        }
        return dataset
    }
}