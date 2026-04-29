package com.csce685.offlinellm

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

/**
 * MainActivity — entry point for the Offline LLM Chat app.
 *
 * Architecture:
 *  MainActivity (Compose UI)
 *      └─ MLCBridge (JNI wrapper)
 *          └─ libmlc_llm.so (MLC-LLM C++ engine)
 *
 * The UI renders a streaming chat interface. Tokens are streamed
 * from the JNI callback and appended to the Compose state flow
 * so the user sees text appear in real-time.
 */
class MainActivity : ComponentActivity() {

    private val TAG = "OfflineLLM"
    private lateinit var bridge: MLCBridge

    // StateFlow drives real-time streaming UI updates
    private val _streamingOutput = MutableStateFlow("")
    val streamingOutput: StateFlow<String> = _streamingOutput

    private val _isLoading = MutableStateFlow(true)
    val isLoading: StateFlow<Boolean> = _isLoading

    private val _perfMetrics = MutableStateFlow(PerfMetrics())
    val perfMetrics: StateFlow<PerfMetrics> = _perfMetrics

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            OfflineLLMTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    ChatScreen(
                        isLoading = isLoading.collectAsState().value,
                        streamingOutput = streamingOutput.collectAsState().value,
                        perfMetrics = perfMetrics.collectAsState().value,
                        onSendMessage = { prompt -> sendMessage(prompt) }
                    )
                }
            }
        }

        initModel()
    }

    private fun initModel() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                bridge = MLCBridge()
                // Model weights are bundled in assets/mlc_models/
                val modelDir = applicationContext.filesDir.absolutePath + "/phi-1_5-q4f16_1-MLC"
                bridge.loadModel(modelDir)
                _isLoading.value = false
                Log.i(TAG, "Model loaded successfully from $modelDir")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model: ${e.message}")
                _isLoading.value = false
            }
        }
    }

    fun sendMessage(prompt: String) {
        lifecycleScope.launch(Dispatchers.IO) {
            _streamingOutput.value = ""
            val startTime = System.currentTimeMillis()
            var firstTokenTime: Long? = null
            var tokenCount = 0

            bridge.chat(
                prompt = prompt,
                maxNewTokens = 256,
                onToken = { token ->
                    if (firstTokenTime == null) firstTokenTime = System.currentTimeMillis()
                    tokenCount++
                    _streamingOutput.value += token
                },
                onFinish = {
                    val totalTime = System.currentTimeMillis() - startTime
                    val ttft = firstTokenTime?.let { it - startTime } ?: 0L
                    val decodeTime = if (firstTokenTime != null) totalTime - ttft else totalTime
                    val tps = if (decodeTime > 0 && tokenCount > 1)
                        ((tokenCount - 1).toDouble() / decodeTime) * 1000.0
                    else 0.0

                    _perfMetrics.value = PerfMetrics(
                        ttftMs = ttft,
                        tps = tps,
                        tokenCount = tokenCount,
                        totalTimeMs = totalTime
                    )
                }
            )
        }
    }
}

data class PerfMetrics(
    val ttftMs: Long = 0L,
    val tps: Double = 0.0,
    val tokenCount: Int = 0,
    val totalTimeMs: Long = 0L
)
