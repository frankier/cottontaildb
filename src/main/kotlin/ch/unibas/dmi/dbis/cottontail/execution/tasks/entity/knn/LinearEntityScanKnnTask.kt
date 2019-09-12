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
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.reduce3.EuclideanDistance
import org.nd4j.linalg.factory.Nd4j

/**
 * A [Task] that executes a sequential boolean kNN on a float [Column] of the specified [Entity].
 *
 * @author Ralph Gasser
 * @version 1.0
 */
internal class LinearEntityScanKnnTask(val entity: Entity, val knn: KnnPredicate<INDArray>, val predicate: BooleanPredicate? = null) : ExecutionTask("LinearEntityScanKnnTask[${entity.fqn}][${knn.column.name}][${knn.distance::class.simpleName}][${knn.k}][q=${knn.query.hashCode()}]") {

    /** Set containing the kNN values. */
    private val knnSet = knn.query.map { HeapSelect<ComparablePair<Long,Double>>(this.knn.k) }

    /** List of the [ColumnDef] this instance of [LinearEntityScanKnnTask] produces. */
    private val produces: Array<ColumnDef<*>> = arrayOf(ColumnDef("${entity.fqn}.distance", ColumnType.forName("DOUBLE")))

    /** The cost of this [LinearEntityScanKnnTask] is constant */
    override val cost = this.entity.statistics.columns * (this.knn.operations * 1e-5 + (this.predicate?.operations ?: 0) * 1e-5).toFloat()

    /**
     * Executes this [LinearEntityScanKnnTask]
     */
    override fun execute(): Recordset {
        /* Extract the necessary data. */
        val columns = arrayOf<ColumnDef<*>>(this.knn.column).plus(predicate?.columns?.toTypedArray() ?: emptyArray())

        /* Execute kNN lookup. */
        this.entity.Tx(readonly = true, columns = columns).begin { tx ->

            val query = Nd4j.createUninitialized(DataType.FLOAT, this.knn.query.first().shape()[0])
            val value = Nd4j.createUninitialized(DataType.FLOAT, this.knn.query.first().shape()[0])
            val op = EuclideanDistance(query, value)
            val exec = Nd4j.getExecutioner()

            tx.forEach {
                if (this.predicate == null || this.predicate.matches(it)) {
                    val v = it[this.knn.column]
                    if (v != null) {
                        this.knn.query.forEachIndexed { i, q ->
                            value.assign(v.value)
                            query.assign(q)
                            this.knnSet[i].add(ComparablePair(it.tupleId, exec.execAndReturn(op).finalResult.toDouble()))
                        }
                    }
                }
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