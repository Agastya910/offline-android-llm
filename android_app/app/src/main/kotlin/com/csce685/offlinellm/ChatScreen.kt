package com.csce685.offlinellm

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * ChatScreen — main Compose UI for the offline LLM chat interface.
 *
 * Features:
 *  - Streaming text output with real-time token display
 *  - Performance overlay (TTFT, TPS)
 *  - Loading indicator while model initialises
 *  - Reset button to clear KV cache and conversation
 */
@Composable
fun ChatScreen(
    isLoading: Boolean,
    streamingOutput: String,
    perfMetrics: PerfMetrics,
    onSendMessage: (String) -> Unit
) {
    var inputText by remember { mutableStateOf("") }
    val listState = rememberLazyListState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(12.dp)
    ) {
        // Header
        Text(
            text = "Offline LLM — Phi-1.5 (q4f16_1)",
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        // Performance overlay
        if (perfMetrics.tokenCount > 0) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(6.dp))
                    .padding(horizontal = 10.dp, vertical = 4.dp),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("TTFT: ${perfMetrics.ttftMs} ms", fontSize = 12.sp)
                Text("TPS: ${"%.1f".format(perfMetrics.tps)}", fontSize = 12.sp)
                Text("Tokens: ${perfMetrics.tokenCount}", fontSize = 12.sp)
            }
            Spacer(modifier = Modifier.height(6.dp))
        }

        // Output area
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(8.dp))
                .padding(10.dp)
        ) {
            if (isLoading) {
                Column(modifier = Modifier.align(Alignment.Center)) {
                    CircularProgressIndicator()
                    Text("Loading model…", modifier = Modifier.padding(top = 8.dp))
                }
            } else {
                Text(
                    text = streamingOutput.ifEmpty { "Enter a prompt below and tap Send." },
                    fontFamily = FontFamily.Monospace,
                    fontSize = 14.sp
                )
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Input row
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedTextField(
                value = inputText,
                onValueChange = { inputText = it },
                modifier = Modifier.weight(1f),
                placeholder = { Text("Type a message…") },
                maxLines = 4
            )
            Spacer(modifier = Modifier.width(8.dp))
            IconButton(
                onClick = {
                    if (inputText.isNotBlank() && !isLoading) {
                        onSendMessage(inputText)
                        inputText = ""
                    }
                }
            ) {
                Icon(Icons.Filled.Send, contentDescription = "Send")
            }
        }
    }
}

@Composable
fun OfflineLLMTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(),
        content = content
    )
}
