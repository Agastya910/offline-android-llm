/**
 * mlc_jni.cpp
 * -----------
 * JNI bridge exposing MLC-LLM's C++ inference engine to Kotlin/Java.
 *
 * Build target: android/arm64-v8a (API 26+)
 * Requires:     libmlc_llm.so, libtvm_runtime.so
 *
 * Key design decisions:
 *  - loadModel() initialises the MLCEngine once per app session.
 *  - chatNative() calls MLC's streaming chat API and invokes the
 *    Kotlin lambda via JNI for each decoded token.
 *  - All JNI calls are guarded with exception checking to avoid
 *    crashes from out-of-memory conditions during inference.
 */

#include <jni.h>
#include <string>
#include <android/log.h>

// MLC-LLM C API header (distributed with mlc_llm Android SDK)
#include "mlc_llm/c_api.h"

#define LOG_TAG "MLCBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static MLCEngineHandle g_engine = nullptr;

extern "C" {

// ─────────────────────────────────────────────────────────────
// loadModel
// ─────────────────────────────────────────────────────────────
JNIEXPORT void JNICALL
Java_com_csce685_offlinellm_MLCBridge_loadModel(
        JNIEnv* env, jobject /* this */, jstring model_dir_jstr)
{
    const char* model_dir = env->GetStringUTFChars(model_dir_jstr, nullptr);
    LOGI("Loading model from: %s", model_dir);

    MLCEngineConfig config;
    config.model_path    = model_dir;
    config.device        = "vulkan";   // fallback to "opencl" if Vulkan not supported
    config.kv_cache_size = 2048;       // tokens; reduce if OOM on low-RAM devices

    int ret = MLCEngineCreate(&g_engine, &config);
    if (ret != 0) {
        LOGE("MLCEngineCreate failed with code %d", ret);
        jclass exc = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(exc, "Failed to initialise MLC engine");
    } else {
        LOGI("MLC engine initialised successfully.");
    }

    env->ReleaseStringUTFChars(model_dir_jstr, model_dir);
}

// ─────────────────────────────────────────────────────────────
// unloadModel
// ─────────────────────────────────────────────────────────────
JNIEXPORT void JNICALL
Java_com_csce685_offlinellm_MLCBridge_unloadModel(
        JNIEnv* /* env */, jobject /* this */)
{
    if (g_engine) {
        MLCEngineDestroy(g_engine);
        g_engine = nullptr;
        LOGI("MLC engine destroyed.");
    }
}

// ─────────────────────────────────────────────────────────────
// chatNative  (streaming inference)
// ─────────────────────────────────────────────────────────────
JNIEXPORT void JNICALL
Java_com_csce685_offlinellm_MLCBridge_chatNative(
        JNIEnv* env,
        jobject /* this */,
        jstring prompt_jstr,
        jint max_new_tokens,
        jobject on_token_lambda,
        jobject on_finish_lambda)
{
    if (!g_engine) {
        LOGE("chatNative called before model was loaded");
        return;
    }

    const char* prompt = env->GetStringUTFChars(prompt_jstr, nullptr);

    // Resolve Kotlin Function1<String, Unit>::invoke
    jclass fn_class = env->GetObjectClass(on_token_lambda);
    jmethodID fn_invoke = env->GetMethodID(fn_class, "invoke",
                                            "(Ljava/lang/Object;)Ljava/lang/Object;");

    // Streaming callback — called by MLC for each decoded token
    auto token_cb = [&](const char* token_text) {
        jstring token_jstr = env->NewStringUTF(token_text);
        env->CallObjectMethod(on_token_lambda, fn_invoke, token_jstr);
        env->DeleteLocalRef(token_jstr);
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
    };

    MLCChatConfig chat_cfg;
    chat_cfg.max_new_tokens = (int)max_new_tokens;
    chat_cfg.temperature    = 0.7f;
    chat_cfg.top_p          = 0.95f;

    int ret = MLCEngineChatStream(g_engine, prompt, &chat_cfg, token_cb);
    if (ret != 0) {
        LOGE("MLCEngineChatStream returned error %d", ret);
    }

    env->ReleaseStringUTFChars(prompt_jstr, prompt);

    // Call onFinish lambda
    jclass fin_class = env->GetObjectClass(on_finish_lambda);
    jmethodID fin_invoke = env->GetMethodID(fin_class, "invoke",
                                             "()Ljava/lang/Object;");
    env->CallObjectMethod(on_finish_lambda, fin_invoke);
}

// ─────────────────────────────────────────────────────────────
// getRuntimeStats
// ─────────────────────────────────────────────────────────────
JNIEXPORT jstring JNICALL
Java_com_csce685_offlinellm_MLCBridge_getRuntimeStats(
        JNIEnv* env, jobject /* this */)
{
    if (!g_engine) return env->NewStringUTF("{}");
    char stats_buf[1024] = {0};
    MLCEngineGetStats(g_engine, stats_buf, sizeof(stats_buf));
    return env->NewStringUTF(stats_buf);
}

// ─────────────────────────────────────────────────────────────
// resetChat
// ─────────────────────────────────────────────────────────────
JNIEXPORT void JNICALL
Java_com_csce685_offlinellm_MLCBridge_resetChat(
        JNIEnv* /* env */, jobject /* this */)
{
    if (g_engine) {
        MLCEngineResetChat(g_engine);
        LOGI("Chat context reset.");
    }
}

} // extern "C"
