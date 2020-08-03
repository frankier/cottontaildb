package org.vitrivr.cottontail.server.grpc.services

import io.grpc.Status
import io.grpc.stub.StreamObserver
import org.slf4j.LoggerFactory
import org.vitrivr.cottontail.database.catalogue.Catalogue
import org.vitrivr.cottontail.database.column.ColumnType
import org.vitrivr.cottontail.database.index.IndexType
import org.vitrivr.cottontail.grpc.CottonDDLGrpc
import org.vitrivr.cottontail.grpc.CottontailGrpc
import org.vitrivr.cottontail.model.basics.ColumnDef
import org.vitrivr.cottontail.model.exceptions.DatabaseException
import org.vitrivr.cottontail.server.grpc.helper.fqn

/**
 * This is a gRPC service endpoint that handles DDL (=Data Definition Language) request for Cottontail DB.
 *
 * @author Ralph Gasser
 * @version 1.1
 */
class CottonDDLService(val catalogue: Catalogue) : CottonDDLGrpc.CottonDDLImplBase() {
    /** Logger used for logging the output. */
    companion object {
        private val LOGGER = LoggerFactory.getLogger(CottonDDLService::class.java)
    }

    /**
     * gRPC endpoint for creating a new [Schema][org.vitrivr.cottontail.database.schema.Schema]
     */
    override fun createSchema(request: CottontailGrpc.Schema, responseObserver: StreamObserver<CottontailGrpc.SuccessStatus>) = try {
        val schemaName = request.fqn()
        LOGGER.trace("Creating schema {}", schemaName)
        this.catalogue.createSchema(schemaName)
        responseObserver.onNext(CottontailGrpc.SuccessStatus.newBuilder().setTimestamp(System.currentTimeMillis()).build())
        responseObserver.onCompleted()
    } catch (e: DatabaseException.SchemaAlreadyExistsException) {
        LOGGER.error("Error while creating schema", e)
        responseObserver.onError(Status.ALREADY_EXISTS.withDescription("Schema '${request.name}' cannot be created because it already exists!").asException())
    } catch (e: DatabaseException) {
        LOGGER.error("Error while creating schema", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to create schema '${request.name}' because of database error: ${e.message}").asException())
    } catch (e: Throwable) {
        LOGGER.error("Error while creating schema", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to create schema '${request.name}' 'because unknown error: ${e.message}").asException())
    }

    /**
     * gRPC endpoint for dropping a [Schema][org.vitrivr.cottontail.database.schema.Schema]
     */
    override fun dropSchema(request: CottontailGrpc.Schema, responseObserver: StreamObserver<CottontailGrpc.SuccessStatus>) = try {
        val schemaName = request.fqn()
        LOGGER.trace("Dropping schema {}", schemaName)
        this.catalogue.dropSchema(schemaName)
        responseObserver.onNext(CottontailGrpc.SuccessStatus.newBuilder().setTimestamp(System.currentTimeMillis()).build())
        responseObserver.onCompleted()
    } catch (e: DatabaseException.SchemaDoesNotExistException) {
        LOGGER.error("Error while dropping schema '${request.name}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Schema '${request.name}' does not exist!").asException())
    } catch (e: DatabaseException) {
        LOGGER.error("Error while dropping schema '${request.name}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to drop schema '${request.name}' because of database error: ${e.message}").asException())
    } catch (e: Throwable) {
        LOGGER.error("Error while dropping schema '${request.name}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to drop schema '${request.name}' because unknown error: ${e.message}").asException())
    }

    /**
     * gRPC endpoint listing the available [Schema][org.vitrivr.cottontail.database.schema.Schema]s.
     */
    override fun listSchemas(request: CottontailGrpc.Empty, responseObserver: StreamObserver<CottontailGrpc.Schema>) = try {
        this.catalogue.schemas.forEach {
            responseObserver.onNext(CottontailGrpc.Schema.newBuilder().setName(it.simple).build())
        }
        responseObserver.onCompleted()
    } catch (e: DatabaseException) {
        LOGGER.error("Error while listing schemas", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to list schemas because of database error: ${e.message}").asException())
    } catch (e: Throwable) {
        LOGGER.error("Error while listing schemas", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to list schemas because of unknown error: ${e.message}").asException())
    }

    /**
     *
     * gRPC endpoint for creating a new [Entity][org.vitrivr.cottontail.database.entity.Entity]
     */
    override fun createEntity(request: CottontailGrpc.EntityDefinition, responseObserver: StreamObserver<CottontailGrpc.SuccessStatus>) = try {
        val entityName = request.entity.fqn()
        LOGGER.trace("Creating entity {}...", entityName)
        val schema = this.catalogue.schemaForName(entityName.schema())
        val columns = request.columnsList.map {
            val type = ColumnType.forName(it.type.name)
            val name = entityName.column(it.name)
            ColumnDef(name, type, it.length, it.nullable)
        }
        schema.createEntity(entityName, *columns.toTypedArray())
        responseObserver.onNext(CottontailGrpc.SuccessStatus.newBuilder().setTimestamp(System.currentTimeMillis()).build())
        responseObserver.onCompleted()
    } catch (e: DatabaseException.SchemaDoesNotExistException) {
        LOGGER.error("Error while creating entity '${request.entity.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Schema '${request.entity.schema.name} does not exist!").asException())
    } catch (e: DatabaseException.EntityAlreadyExistsException) {
        LOGGER.error("Error while creating entity '${request.entity.fqn()}'", e)
        responseObserver.onError(Status.ALREADY_EXISTS.withDescription("Entity '${request.entity.fqn()} does already exist!").asException())
    } catch (e: DatabaseException) {
        LOGGER.error("Error while creating entity '${request.entity.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to create entity '${request.entity.fqn()}' because of database error: ${e.message}").asException())
    } catch (e: Throwable) {
        LOGGER.error("Error while creating entity '${request.entity.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to create entity '${request.entity.fqn()}' because of unknown error: ${e.message}").asException())
    }

    override fun entityDetails(request: CottontailGrpc.Entity, responseObserver: StreamObserver<CottontailGrpc.EntityDefinition>) = try {
        val entityName = request.fqn()
        val entity = this.catalogue.schemaForName(entityName.schema()).entityForName(entityName)
        val def = CottontailGrpc.EntityDefinition.newBuilder().setEntity(request)
        for (c in entity.allColumns()) {
            def.addColumns(CottontailGrpc.ColumnDefinition.newBuilder().setName(c.name.simple).setNullable(c.nullable).setLength(c.logicalSize).setType(CottontailGrpc.Type.valueOf(c.type.name)))
        }
        responseObserver.onNext(def.build())
        responseObserver.onCompleted()
    } catch (e: DatabaseException.SchemaDoesNotExistException) {
        LOGGER.error("Error while fetching information for entity '${request.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Schema '${request.schema.fqn()}' does not exist!").asException())
    } catch (e: DatabaseException.EntityDoesNotExistException) {
        LOGGER.error("Error while fetching information for entity '${request.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Entity '${request.fqn()}' does not exist!").asException())
    } catch (e: DatabaseException) {
        LOGGER.error("Error while fetching information for entity '${request.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to drop entity '${request.fqn()}' because of database error: ${e.message}").asException())
    } catch (e: Throwable) {
        LOGGER.error("Error while fetching information for entity '${request.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to drop entity '${request.fqn()}' because of unknown error: ${e.message}").asException())
    }

    /**
     * gRPC endpoint for dropping a particular [Schema][org.vitrivr.cottontail.database.schema.Schema]
     */
    override fun dropEntity(request: CottontailGrpc.Entity, responseObserver: StreamObserver<CottontailGrpc.SuccessStatus>) = try {
        val entityName = request.fqn()
        LOGGER.trace("Dropping entity {}...", entityName)
        this.catalogue.schemaForName(entityName.schema()).dropEntity(entityName)
        responseObserver.onNext(CottontailGrpc.SuccessStatus.newBuilder().setTimestamp(System.currentTimeMillis()).build())
        responseObserver.onCompleted()
    } catch (e: DatabaseException.SchemaDoesNotExistException) {
        LOGGER.error("Error while dropping entity '${request.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Schema '${request.schema.fqn()}' does not exist!").asException())
    } catch (e: DatabaseException.EntityDoesNotExistException) {
        LOGGER.error("Error while dropping entity '${request.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Entity '${request.fqn()}' does not exist!").asException())
    } catch (e: DatabaseException) {
        LOGGER.error("Error while dropping entity '${request.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to drop entity '${request.fqn()}' because of database error: ${e.message}").asException())
    } catch (e: Throwable) {
        LOGGER.error("Error while dropping entity '${request.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to drop entity '${request.fqn()}' because of unknown error: ${e.message}").asException())
    }

    /**
     * gRPC endpoint listing the available [Entity][org.vitrivr.cottontail.database.entity.Entity]s
     * for the provided [Schema][org.vitrivr.cottontail.database.schema.Schema].
     */
    override fun listEntities(request: CottontailGrpc.Schema, responseObserver: StreamObserver<CottontailGrpc.Entity>) = try {
        val schemaName = request.fqn()
        val builder = CottontailGrpc.Entity.newBuilder()
        this.catalogue.schemaForName(schemaName).entities.forEach {
            responseObserver.onNext(builder.setName(it.simple).setSchema(request).build())
        }
        responseObserver.onCompleted()
    } catch (e: DatabaseException.SchemaDoesNotExistException) {
        LOGGER.error("Error while listing entities", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Schema '${request.name} does not exist!").asException())
    } catch (e: DatabaseException) {
        LOGGER.error("Error while listing entities", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to list entities for schema ${request.name} because of database error: ${e.message}").asException())
    } catch (e: Throwable) {
        LOGGER.error("Error while listing entities", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to list entities for schema ${request.name} because of unknown error: ${e.message}").asException())
    }

    /**
     * gRPC endpoint for creating a particular [Index][org.vitrivr.cottontail.database.index.Index]
     */
    override fun createIndex(request: CottontailGrpc.IndexDefinition, responseObserver: StreamObserver<CottontailGrpc.SuccessStatus>) = try {
        LOGGER.trace("Creating index {}", request)
        val indexName = request.index.fqn()
        val entity = this.catalogue.schemaForName(indexName.schema()).entityForName(indexName.entity())
        val columns = request.columnsList.map {
            val columnName = indexName.entity().column(it)
            entity.columnForName(columnName)
                    ?: throw DatabaseException.ColumnDoesNotExistException(columnName)
        }.toTypedArray()

        /* Creates and updates the index. */
        entity.createIndex(indexName, IndexType.valueOf(request.index.type.toString()), columns, request.paramsMap)

        /* Notify caller of success. */
        responseObserver.onNext(CottontailGrpc.SuccessStatus.newBuilder().setTimestamp(System.currentTimeMillis()).build())
        responseObserver.onCompleted()
        LOGGER.trace("Index {} created successfully!", request)
    } catch (e: DatabaseException.SchemaDoesNotExistException) {
        LOGGER.error("Error while creating index '${request.index.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Schema '${request.index.entity.schema.fqn()} does not exist!").asException())
    } catch (e: DatabaseException.EntityDoesNotExistException) {
        LOGGER.error("Error while creating index '${request.index.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Entity '${request.index.entity.fqn()} does not exist!").asException())
    } catch (e: DatabaseException.IndexAlreadyExistsException) {
        LOGGER.error("Error while creating index '${request.index.fqn()}'", e)
        responseObserver.onError(Status.ALREADY_EXISTS.withDescription("Index '${request.index.fqn()}' does already exist!").asException())
    } catch (e: DatabaseException.ColumnDoesNotExistException) {
        LOGGER.error("Error while creating index '${request.index.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription(e.message).asException())
    } catch (e: DatabaseException) {
        LOGGER.error("Error while creating index '${request.index.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to create index '${request.index.fqn()}' because of database error: ${e.message}").asException())
    } catch (e: Throwable) {
        LOGGER.error("Error while creating index '${request.index.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to create index '${request.index.fqn()}' because of an unknown error: ${e.message}").asException())
    }

    /**
     * gRPC endpoint for dropping a particular [Index][org.vitrivr.cottontail.database.index.Index]
     */
    override fun dropIndex(request: CottontailGrpc.Index, responseObserver: StreamObserver<CottontailGrpc.SuccessStatus>) = try {
        val indexName = request.fqn()
        LOGGER.trace("Dropping index {}", indexName)
        this.catalogue.schemaForName(indexName.schema()).entityForName(indexName.entity()).dropIndex(indexName)

        /* Notify caller of success. */
        responseObserver.onNext(CottontailGrpc.SuccessStatus.newBuilder().setTimestamp(System.currentTimeMillis()).build())
        responseObserver.onCompleted()
        LOGGER.trace("Index {} dropped successfully!", request)
    } catch (e: DatabaseException.SchemaDoesNotExistException) {
        LOGGER.error("Error while dropping index '${request.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Schema '${request.entity.schema.fqn()} does not exist!").asException())
    } catch (e: DatabaseException.EntityDoesNotExistException) {
        LOGGER.error("Error while dropping index '${request.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Entity '${request.entity.fqn()} does not exist!").asException())
    } catch (e: DatabaseException.IndexDoesNotExistException) {
        LOGGER.error("Error while dropping index '${request.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Index '${request.fqn()} does not exist!").asException())
    } catch (e: DatabaseException) {
        LOGGER.error("Error while dropping index '${request.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to drop index '${request.fqn()}' because of database error: ${e.message}").asException())
    } catch (e: Throwable) {
        LOGGER.error("Error while dropping index '${request.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to drop index '${request.fqn()}' because of an unknown error: ${e.message}").asException())
    }

    /**
     * gRPC endpoint for rebuilding a particular [Index][org.vitrivr.cottontail.database.index.Index]
     */
    override fun rebuildIndex(request: CottontailGrpc.Index, responseObserver: StreamObserver<CottontailGrpc.SuccessStatus>) = try {
        val indexName = request.fqn()
        LOGGER.trace("Rebuilding index {}", indexName)

        /* Update index. */
        this.catalogue.schemaForName(indexName.schema()).entityForName(indexName.entity()).updateIndex(indexName)

        /* Notify caller of success. */
        responseObserver.onNext(CottontailGrpc.SuccessStatus.newBuilder().setTimestamp(System.currentTimeMillis()).build())
        responseObserver.onCompleted()
        LOGGER.trace("Index {} rebuilt successfully!", request)
    } catch (e: DatabaseException.SchemaDoesNotExistException) {
        LOGGER.error("Error while rebuilding index '${request.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Schema '${request.entity.schema.fqn()} does not exist!").asException())
    } catch (e: DatabaseException.EntityDoesNotExistException) {
        LOGGER.error("Error while rebuilding index '${request.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Entity '${request.entity.fqn()} does not exist!").asException())
    } catch (e: DatabaseException.IndexDoesNotExistException) {
        LOGGER.error("Error while rebuilding index '${request.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Index '${request.fqn()} does not exist!").asException())
    } catch (e: DatabaseException) {
        LOGGER.error("Error while rebuilding index '${request.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to rebuild index '${request.fqn()}' because of database error: ${e.message}").asException())
    } catch (e: Throwable) {
        LOGGER.error("Error while rebuilding index '${request.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to rebuild index '${request.fqn()}' because of an unknown error: ${e.message}").asException())
    }

    /**
     * gRPC endpoint for listing available [Index][org.vitrivr.cottontail.database.index.Index] for a given [Entity][org.vitrivr.cottontail.database.entity.Entity]
     */
    override fun listIndexes(request: CottontailGrpc.Entity, responseObserver: StreamObserver<CottontailGrpc.Index>) = try {
        val entityName = request.fqn()

        /* Get entity and stream available index structures. */
        val entity = this.catalogue.schemaForName(entityName.schema()).entityForName(entityName)
        entity.allIndexes().forEach {
            val index = CottontailGrpc.Index.newBuilder()
                    .setName(it.name.simple)
                    .setType(CottontailGrpc.IndexType.valueOf(it.type.name))
                    .setEntity(CottontailGrpc.Entity.newBuilder().setName(entity.name.simple).setSchema(CottontailGrpc.Schema.newBuilder().setName(entity.parent.name.simple)))
            responseObserver.onNext(index.build())
        }

        /* Notify caller of success. */
        responseObserver.onCompleted()

    } catch (e: DatabaseException.SchemaDoesNotExistException) {
        LOGGER.error("Error while optimizing entity '${request.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Schema '${request.schema.fqn()} does not exist!").asException())
    } catch (e: DatabaseException.EntityDoesNotExistException) {
        LOGGER.error("Error while optimizing entity '${request.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Entity '${request.fqn()} does not exist!").asException())
    } catch (e: DatabaseException) {
        LOGGER.error("Error while optimizing entity '${request.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to optimize entity '${request.fqn()}' because of database error: ${e.message}").asException())
    } catch (e: Throwable) {
        LOGGER.error("Error while optimizing entity '${request.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to optimize entity '${request.fqn()}' because of an unknown error: ${e.message}").asException())
    }

    /**
     * gRPC endpoint for optimizing a particular entity. Currently just rebuilds all the indexes.
     */
    override fun optimizeEntity(request: CottontailGrpc.Entity, responseObserver: StreamObserver<CottontailGrpc.SuccessStatus>) = try {
        val entityName = request.fqn()

        /* Update indexes. */
        this.catalogue.schemaForName(entityName.schema()).entityForName(entityName).updateAllIndexes()

        /* Notify caller of success. */
        responseObserver.onNext(CottontailGrpc.SuccessStatus.newBuilder().setTimestamp(System.currentTimeMillis()).build())
        responseObserver.onCompleted()
    } catch (e: DatabaseException.SchemaDoesNotExistException) {
        LOGGER.error("Error while optimizing entity '${request.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Schema '${request.schema.fqn()} does not exist!").asException())
    } catch (e: DatabaseException.EntityDoesNotExistException) {
        LOGGER.error("Error while optimizing entity '${request.fqn()}'", e)
        responseObserver.onError(Status.NOT_FOUND.withDescription("Entity '${request.fqn()} does not exist!").asException())
    } catch (e: DatabaseException) {
        LOGGER.error("Error while optimizing entity '${request.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to optimize entity '${request.fqn()}' because of database error: ${e.message}").asException())
    } catch (e: Throwable) {
        LOGGER.error("Error while optimizing entity '${request.fqn()}'", e)
        responseObserver.onError(Status.UNKNOWN.withDescription("Failed to optimize entity '${request.fqn()}' because of an unknown error: ${e.message}").asException())
    }
}
