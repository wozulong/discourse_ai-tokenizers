# frozen_string_literal: true

module DiscourseAi
  module Tokenizer
    # Wrapper for OpenAI tokenizer library for compatibility with Discourse AI API
    class OpenAiTokenizer < BasicTokenizer
      class << self
        def tokenizer
          @tokenizer ||= Tiktoken.get_encoding("o200k_base")
        end

        def tokenize(text)
          tokenizer.encode(text)
        end

        def encode(text)
          tokenizer.encode(text)
        end

        def decode(token_ids)
          tokenizer.decode(token_ids)
        rescue Tiktoken::UnicodeError => e
          # Handle invalid token IDs gracefully by returning empty string
          ""
        end

        def truncate(text, max_length, strict: false)
          return "" if max_length <= 0

          # fast track common case, /2 to handle unicode chars
          # than can take more than 1 token per char
          return text if !strict && text.size < max_length / 2
          
          # Take tokens up to max_length, decode, then ensure we don't exceed limit
          tokens = tokenize(text)
          return text if tokens.length <= max_length
          
          # Take tokens up to max_length
          truncated_tokens = tokens.take(max_length)
          
          # Try to decode, with retry on Unicode errors
          begin
            tokenizer.decode(truncated_tokens)
          rescue Tiktoken::UnicodeError
            truncated_tokens.pop
            retry unless truncated_tokens.empty?
            ""
          end
        end

        def below_limit?(text, limit, strict: false)
          # fast track common case, /2 to handle unicode chars
          # than can take more than 1 token per char
          return true if !strict && text.size < limit / 2

          tokenizer.encode(text).length < limit
        end
      end
    end

    OpenAiO200kTokenizer = OpenAiTokenizer
  end
end
