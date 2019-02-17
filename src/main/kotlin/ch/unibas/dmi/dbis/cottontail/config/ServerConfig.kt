package ch.unibas.dmi.dbis.cottontail.config

import ch.unibas.dmi.dbis.cottontail.utilities.serializers.PathSerializer
import kotlinx.serialization.Optional
import kotlinx.serialization.Serializable
import java.nio.file.Path

/**
 * Config for Cottontail DB's GRPC server.
 *
 *
 * @param port The port under which Cottontail DB demon should be listening for calls. Defaults to 1865
 * @param messageSize The the maximum size an incoming GRPC message can have. Defaults to 524'288 bytes (512 kbytes).
 * @param coreThreads The core size of the thread pool used to run the GRPC server.
 * @param maxThreads The maximum number of threads that handle calls to the GRPC server.
 * @param keepAliveTime The number of milliseconds to wait before decommissioning unused threads.
 * @param certFile Path to the certificate file used for TLS.
 * @param privateKey Path to the private key used for TLS.
 *
 * @author Ralph Gasser
 * @version 1.0
 */
@Serializable
data class ServerConfig(
    @Optional val port: Int = 1865,
    @Optional val messageSize: Int = 524288,
    @Optional val coreThreads: Int = Runtime.getRuntime().availableProcessors() / 2,
    @Optional val maxThreads: Int = Runtime.getRuntime().availableProcessors() * 2,
    @Optional val keepAliveTime: Long = 500,
    @Optional @Serializable(with=PathSerializer::class) val certFile: Path? = null,
    @Optional @Serializable(with=PathSerializer::class) val privateKey: Path? = null) {

    /**
     * True if TLS should be used for GRPC communication, false otherwise.
     */
    val useTls
        get() = this.certFile != null && this.privateKey != null

}


