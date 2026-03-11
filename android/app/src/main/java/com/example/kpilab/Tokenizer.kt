package com.example.kpilab

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.nio.LongBuffer

/**
 * CLIP BPE Tokenizer for SD v1.5 Text Encoder.
 * Loads vocab.json + merges.txt from assets.
 * Produces token IDs as INT64 [1, 77] with BOS/EOS/PAD.
 */
class Tokenizer(context: Context) {

    companion object {
        private const val TAG = "Tokenizer"
        private const val MAX_LENGTH = 77
        private const val BOS_TOKEN = 49406  // <|startoftext|>
        private const val EOS_TOKEN = 49407  // <|endoftext|>
        private const val PAD_TOKEN = 49407  // Same as EOS for CLIP
    }

    private val vocab: Map<String, Int>
    private val merges: List<Pair<String, String>>

    init {
        // Load vocabulary
        val vocabJson = context.assets.open("vocab.json").bufferedReader().use { it.readText() }
        val jsonObj = JSONObject(vocabJson)
        val vocabMap = mutableMapOf<String, Int>()
        jsonObj.keys().forEach { key ->
            vocabMap[key] = jsonObj.getInt(key)
        }
        vocab = vocabMap
        Log.i(TAG, "Vocab loaded: ${vocab.size} tokens")

        // Load BPE merges
        val mergesText = context.assets.open("merges.txt").bufferedReader().use { it.readLines() }
        merges = mergesText
            .drop(1) // Skip header line
            .filter { it.isNotBlank() }
            .map {
                val parts = it.split(" ")
                Pair(parts[0], parts[1])
            }
        Log.i(TAG, "Merges loaded: ${merges.size} rules")
    }

    /**
     * Tokenize text and return as LongBuffer (INT64) for ORT input.
     * CLIP Text Encoder expects input_ids as INT64 tensor.
     * @return Pair of (LongBuffer with token IDs, shape [1, 77])
     */
    fun tokenize(text: String): Pair<LongBuffer, LongArray> {
        val tokens = encode(text)
        val buffer = LongBuffer.allocate(MAX_LENGTH)

        // BOS + tokens + EOS + padding
        buffer.put(0, BOS_TOKEN.toLong())
        val maxTokens = MAX_LENGTH - 2  // Reserve space for BOS and EOS
        val truncated = tokens.take(maxTokens)
        for (i in truncated.indices) {
            buffer.put(i + 1, truncated[i].toLong())
        }
        buffer.put(truncated.size + 1, EOS_TOKEN.toLong())

        // Pad remaining with PAD token
        for (i in (truncated.size + 2) until MAX_LENGTH) {
            buffer.put(i, PAD_TOKEN.toLong())
        }

        val shape = longArrayOf(1, MAX_LENGTH.toLong())
        return Pair(buffer, shape)
    }

    /**
     * BPE encode a text string to token IDs.
     */
    private fun encode(text: String): List<Int> {
        val cleaned = text.lowercase().trim()
        val words = cleaned.split(Regex("\\s+"))
        val allTokenIds = mutableListOf<Int>()

        for (word in words) {
            // Add </w> suffix for CLIP tokenizer convention
            val wordChars = word.map { it.toString() }.toMutableList()
            if (wordChars.isNotEmpty()) {
                wordChars[wordChars.size - 1] = wordChars.last() + "</w>"
            }

            var pairs = wordChars.toMutableList()
            pairs = applyBpeMerges(pairs)

            for (token in pairs) {
                val id = vocab[token]
                if (id != null) {
                    allTokenIds.add(id)
                } else {
                    // Fallback: encode character by character
                    for (ch in token) {
                        val chId = vocab[ch.toString()]
                        if (chId != null) allTokenIds.add(chId)
                    }
                }
            }
        }

        return allTokenIds
    }

    /**
     * Apply BPE merge rules iteratively until no more merges apply.
     */
    private fun applyBpeMerges(tokens: MutableList<String>): MutableList<String> {
        if (tokens.size <= 1) return tokens

        var current = tokens
        val mergeRank = merges.withIndex().associate { (i, pair) -> pair to i }

        while (true) {
            if (current.size <= 1) break

            // Find the highest-priority merge (lowest rank) among adjacent pairs
            var bestPair: Pair<String, String>? = null
            var bestRank = Int.MAX_VALUE

            for (i in 0 until current.size - 1) {
                val pair = Pair(current[i], current[i + 1])
                val rank = mergeRank[pair]
                if (rank != null && rank < bestRank) {
                    bestRank = rank
                    bestPair = pair
                }
            }

            if (bestPair == null) break

            // Apply the merge
            val merged = bestPair.first + bestPair.second
            val newTokens = mutableListOf<String>()
            var i = 0
            while (i < current.size) {
                if (i < current.size - 1 &&
                    current[i] == bestPair.first && current[i + 1] == bestPair.second) {
                    newTokens.add(merged)
                    i += 2
                } else {
                    newTokens.add(current[i])
                    i++
                }
            }
            current = newTokens
        }

        return current
    }
}
