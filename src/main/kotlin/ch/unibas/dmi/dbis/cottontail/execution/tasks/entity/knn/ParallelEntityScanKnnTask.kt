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
import org.nd4j.linalg.ops.transforms.Transforms

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
        val columnSize = this.knn.column.size
        val numberOfQueries = this.knn.query.size
        val batchSize = 4096 * numberOfQueries
        val dataType = this.knn.query.first().dataType()
        /* Prepare query tensor. */
        val query = Nd4j.createUninitialized(dataType, columnSize.toLong(), batchSize.toLong())
        for (i in 0 until batchSize) {
            query.putColumn(i, this.knn.query[i % numberOfQueries])
        }

        /*  Prepare INDArray for query. */
        val exec = Nd4j.getExecutioner()

        /* Execute kNN lookup. */
        this.entity.Tx(readonly = true, columns = columns).begin { tx ->
            runBlocking {
                val jobs = Array(parallelism.toInt()) { j ->
                    GlobalScope.launch {
                        val start = stride * j + 1
                        val end = stride * j + 1 + stride - 1
                        val value = Nd4j.createUninitialized(dataType, columnSize.toLong(), batchSize.toLong())
                        val op = ManhattanDistance(query, value, 0)
                        var idx = 0


                        Transforms
                        tx.forEach(start, end) { it ->
                            if (this@ParallelEntityScanKnnTask.predicate == null || this@ParallelEntityScanKnnTask.predicate.matches(it)) {
                                val v = it[this@ParallelEntityScanKnnTask.knn.column]
                                if (v != null) {
                                    for (i in 0 until numberOfQueries) {
                                        value.putColumn(idx++, v.value)
                                    }
                                    if (idx == batchSize) {
                                        val result = exec.exec(op)
                                        for (i in 0 until batchSize) {
                                            this@ParallelEntityScanKnnTask.knnSet[i % numberOfQueries].add(ComparablePair(it.tupleId, result.getDouble(i)))
                                        }
                                        idx = 0
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