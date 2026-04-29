package com.csce685.offlinellm

/**
 * MLCBridge — Kotlin JNI wrapper around the MLC-LLM C++ inference engine.
 *
 * The native library (libmlc_llm.so + libtvm_runtime.so) must be compiled
 * for android/arm64-v8a and placed in app/src/main/jniLibs/arm64-v8a/.
 *
 * Native methods are implemented in jni/mlc_jni.cpp.
 */
class MLCBridge {

    init {
        System.loadLibrary("tvm_runtime")
        System.loadLibrary("mlc_llm")
    }

    /**
     * Load the model from disk. This initialises the TVM runtime,
     * allocates KV-cache, and warms up the GPU.
     *
     * @param modelDir absolute path to the MLC model directory on-device
     */
    external fun loadModel(modelDir: String)

    /**
     * Unload the model and free all GPU/CPU memory.
     */
    external fun unloadModel()

    /**
     * Run streaming chat inference.
     *
     * @param prompt        the user prompt string
     * @param maxNewTokens  maximum output tokens
     * @param onToken       called for each decoded token (may be called from a native thread)
     * @param onFinish      called once generation is complete
     */
    fun chat(
        prompt: String,
        maxNewTokens: Int = 256,
        onToken: (String) -> Unit,
        onFinish: () -> Unit
    ) {
        chatNative(prompt, maxNewTokens, onToken, onFinish)
    }

    private external fun chatNative(
        prompt: String,
        maxNewTokens: Int,
        onToken: (String) -> Unit,
        onFinish: () -> Unit
    )

    /**
     * Returns a JSON string with runtime diagnostics:
     * prefill TPS, decode TPS, KV-cache usage, and GPU memory.
     */
    external fun getRuntimeStats(): String

    /**
     * Reset conversation history (clears KV cache).
     */
    external fun resetChat()
}
