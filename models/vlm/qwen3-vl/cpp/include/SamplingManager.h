/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: SamplingManager.h
 * Description:
 *   Sampling manager for text generation post-processing.
 *   Implements temperature, top-k, top-p, presence penalty,
 *   and repetition penalty sampling.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __SAMPLING_MANAGER_H__
#define __SAMPLING_MANAGER_H__

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <numeric>
#include <unordered_map>
#include <vector>

/**
 * @brief Sampling manager for post-processing logits in text generation
 *
 * This class implements various sampling strategies including:
 * - Temperature scaling
 * - Top-K filtering
 * - Top-P (nucleus) filtering
 * - Presence penalty
 * - N-gram repetition blocking (optional, disabled by default)
 */
class SamplingManager {
 public:
  /**
   * @brief Construct a new Sampling Manager object
   *
   * @param temperature Temperature for scaling logits (>0)
   * @param top_k Number of top tokens to keep (-1 or 0 means disabled)
   * @param top_p Cumulative probability threshold for nucleus sampling (0-1)
   * @param repetition_penalty Multiplicative penalty for repeated tokens
   * @param presence_penalty Subtractive presence penalty for repeated tokens
   * @param min_tokens_to_keep Minimum number of tokens to keep in top-p
   */
  SamplingManager(float temperature = 1.0f, int top_k = -1, float top_p = 1.0f,
                  float repetition_penalty = 1.0f,
                  float presence_penalty = 0.0f, int min_tokens_to_keep = 1)
      : temperature_(temperature),
        top_k_(top_k),
        top_p_(top_p),
        min_p_(0.0f),
        presence_penalty_(presence_penalty),
        repetition_penalty_(repetition_penalty),
        min_tokens_to_keep_(min_tokens_to_keep),
        no_repeat_ngram_size_(0),
        repeat_ngram_size_(8),
        repeat_count_threshold_(3),
        repeat_triggered_(false),
        consecutive_trigger_count_(0) {}

  /**
   * @brief Apply softmax to convert logits to probabilities
   *
   * @param logits Input logits array
   * @param size Size of the array
   * @param probs Output probabilities array (must be pre-allocated)
   */
  void softmax(const float* logits, size_t size, float* probs) const;

  /**
   * @brief Apply temperature scaling to logits
   *
   * @param logits Input/output logits array
   * @param size Size of the array
   */
  void applyTemperature(float* logits, size_t size) const;

  /**
   * @brief Apply multiplicative repetition penalty to logits
   *
   * @param logits Input/output logits array
   * @param size Size of the array
   * @param previous_tokens List of previously generated token IDs
   */
  void applyRepetitionPenalty(float* logits, size_t size,
                              const std::vector<int>& previous_tokens) const;

  /**
   * @brief Apply presence penalty (subtractive penalty)
   *
   * @param logits Input/output logits array
   * @param size Size of the array
   * @param previous_tokens List of previously generated token IDs
   */
  void applyPresenceRepetitionPenalty(
      float* logits, size_t size,
      const std::vector<int>& previous_tokens) const;

  /**
   * @brief Apply top-K filtering to probabilities
   *
   * @param probs Input/output probabilities array
   * @param size Size of the array
   */
  void applyTopK(float* probs, size_t size) const;

  /**
   * @brief Apply top-P (nucleus) filtering to probabilities
   *
   * @param probs Input/output probabilities array
   * @param size Size of the array
   */
  void applyTopP(float* probs, size_t size) const;

  /**
   * @brief Process logits through all sampling steps
   *
   * Order: repetition_penalty -> presence_penalty -> top_k -> top_p -> min_p
   * -> temperature
   *
   * @param logits Input logits array
   * @param size Size of the array
   * @param previous_tokens List of previously generated token IDs
   * @param processed_probs Output processed probabilities (must be
   * pre-allocated)
   */
  void processLogits(const float* logits, size_t size,
                     const std::vector<int>& previous_tokens,
                     float* processed_probs) const;

  /**
   * @brief Sample a token from processed logits
   *
   * @param logits Input logits array (shape: [1][vocab_size])
   * @param size Size of the logits array (vocab_size)
   * @param previous_tokens List of previously generated token IDs
   * @return Sampled token ID
   */
  int sample(const float* logits, size_t size,
             const std::vector<int>& previous_tokens) const;

  /**
   * @brief Get processed probabilities without sampling
   *
   * @param logits Input logits array
   * @param size Size of the array
   * @param previous_tokens List of previously generated token IDs
   * @param probs Output probabilities (must be pre-allocated)
   */
  void getProcessedProbs(const float* logits, size_t size,
                         const std::vector<int>& previous_tokens,
                         float* probs) const;

  // Getters
  float getTemperature() const { return temperature_; }
  int getTopK() const { return top_k_; }
  float getTopP() const { return top_p_; }
  float getMinP() const { return min_p_; }
  float getPresencePenalty() const { return presence_penalty_; }
  float getRepetitionPenalty() const { return repetition_penalty_; }
  int getMinTokensToKeep() const { return min_tokens_to_keep_; }

  // N-gram repetition control getters
  int getNoRepeatNgramSize() const { return no_repeat_ngram_size_; }
  int getRepeatNgramSize() const { return repeat_ngram_size_; }
  int getRepeatCountThreshold() const { return repeat_count_threshold_; }
  bool isRepeatTriggered() const { return repeat_triggered_; }
  int getConsecutiveTriggerCount() const { return consecutive_trigger_count_; }
  bool shouldForceStop() const { return consecutive_trigger_count_ >= 10; }

  // Setters
  void setTemperature(float temp) { temperature_ = temp; }
  void setTopK(int k) { top_k_ = k; }
  void setTopP(float p) { top_p_ = p; }
  void setMinP(float p) { min_p_ = p; }
  void setPresencePenalty(float penalty) { presence_penalty_ = penalty; }
  void setRepetitionPenalty(float penalty) { repetition_penalty_ = penalty; }
  void setMinTokensToKeep(int min_keep) { min_tokens_to_keep_ = min_keep; }

  // N-gram repetition control setters
  void setNoRepeatNgramSize(int size) { no_repeat_ngram_size_ = size; }
  void setRepeatNgramSize(int size) { repeat_ngram_size_ = size; }
  void setRepeatCountThreshold(int threshold) {
    repeat_count_threshold_ = threshold;
  }

  /**
   * @brief Reset N-gram pools and state (call at start of new generation)
   */
  void resetNgramState();

  /**
   * @brief Update N-gram pools with current conversation history
   *
   * @param history_tokens All tokens generated in current turn
   */
  void updateNgramPools(const std::vector<int>& history_tokens);

  /**
   * @brief Check if a candidate token would create a repeated N-gram
   *
   * @param candidate_token The candidate token to check
   * @param previous_tokens The previous tokens in the sequence
   * @return true if the candidate would create a repeated N-gram
   */
  bool wouldCreateRepeatedNgram(int candidate_token,
                                const std::vector<int>& previous_tokens) const;

 private:
  float temperature_;         // Temperature for scaling (>0)
  int top_k_;                 // Top-K threshold (-1 or 0 means disabled)
  float top_p_;               // Top-P threshold (0-1, 1.0 means disabled)
  float min_p_;               // Min-P threshold (0.0 means disabled)
  float presence_penalty_;    // Presence penalty (0.0 means disabled)
  float repetition_penalty_;  // Multiplicative repetition penalty
  int min_tokens_to_keep_;    // Minimum tokens to keep in top-p

  // N-gram repetition blocking (disabled by default: no_repeat_ngram_size_ = 0)
  int no_repeat_ngram_size_;    // Size of N-gram to block (0 means disabled)
  int repeat_ngram_size_;       // N-gram size for repeat detection
  int repeat_count_threshold_;  // Threshold to trigger repeat blocking

  // N-gram state (uses full conversation history instead of sliding window)
  std::unordered_map<size_t, int>
      ngram_pool_;  // N-gram -> count (for blocking)
  std::unordered_map<size_t, int> repeat_ngram_pool_;  // Repeat detection pool

  bool repeat_triggered_;  // Whether repeat blocking is currently active
  int consecutive_trigger_count_;  // Consecutive trigger count for strong
                                   // fallback

  /**
   * @brief Hash an N-gram vector to a size_t for use as map key
   */
  size_t hashNgram(const std::vector<int>& ngram) const;
};

#endif  // __SAMPLING_MANAGER_H__