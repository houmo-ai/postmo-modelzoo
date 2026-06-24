#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility> // for std::pair

namespace tokenizer {

// ==========================================
// 1. Data Types
// ==========================================

// Role, Content
using ChatMessage = std::pair<std::string, std::string>;
using ChatMessages = std::vector<ChatMessage>;

// ==========================================
// 2. Main Class (PIMPL Wrapper)
// ==========================================

class PreTrainedTokenizer {
public:
    PreTrainedTokenizer();
    ~PreTrainedTokenizer();

    // Disable copying (PIMPL unique_ptr constraint)
    PreTrainedTokenizer(const PreTrainedTokenizer&) = delete;
    PreTrainedTokenizer& operator=(const PreTrainedTokenizer&) = delete;

    // --- Core API ---
    std::vector<int> encode(const std::string& text, bool add_special_tokens = true) const;
    std::string decode(const std::vector<int>& ids, bool skip_special_tokens = true) const;

    // --- Helpers ---
    int token_to_id(const std::string& token) const;
    std::string id_to_token(int id) const;

    // Special Token Accessors
    int pad_token_id() const;
    int bos_token_id() const;
    int eos_token_id() const;
    int unk_token_id() const;

    // --- Chat Template ---
    void set_chat_template(const std::string& template_str);

    std::string apply_chat_template(
        const ChatMessages& messages,
        bool add_generation_prompt = true
    ) const;

    std::string apply_chat_template(
        const std::string& json_str,
        bool add_generation_prompt = true
    ) const;

    // --- Loading ---
    bool load_from_json_str(const std::string& json_content);

    // --- Configuration ---
    void set_clean_up_tokenization_spaces(bool clean);

private:
    struct Impl; // Forward declaration
    std::unique_ptr<Impl> impl_;
};

// ==========================================
// 3. Factory
// ==========================================
class AutoTokenizer {
public:
    static std::shared_ptr<PreTrainedTokenizer> from_pretrained(const std::string& path);
};

} // namespace tokenizer
