//
//  LlamaBridge.mm
//  原型验证:llama.cpp C API 封装(Objective-C++)
//
//  使用 llama.cpp 最新 API(sampler chain 模式)
//  编译时通过 HEADER_SEARCH_PATHS 指向 xcframework 的 Headers
//

#import "LlamaBridge.h"
#include <llama.h>
#include <string>
#include <vector>

@implementation LlamaModelInfo
@end

@implementation LlamaBridge {
    llama_model * _model;
    llama_context * _ctx;
    llama_sampler * _sampler;
    NSString * _modelPath;
    BOOL _loaded;
}

+ (void)initializeBackend {
    llama_backend_init();
    // 显式检查 Metal
    llama_log_set([](enum ggml_log_level level, const char * text, void * user_data) {
        if (level <= GGML_LOG_LEVEL_WARN) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _model = nullptr;
        _ctx = nullptr;
        _sampler = nullptr;
        _loaded = NO;
    }
    return self;
}

- (void)dealloc {
    [self unloadModel];
}

- (BOOL)loadModelAtPath:(NSString *)path error:(NSError **)error {
    if (_loaded) {
        [self unloadModel];
    }

    llama_model_params modelParams = llama_model_default_params();
    modelParams.n_gpu_layers = 999;        // 全部层放 GPU(Metal)

    _model = llama_model_load_from_file([path UTF8String], modelParams);
    if (!_model) {
        if (error) {
            *error = [NSError errorWithDomain:@"LlamaBridge"
                                        code:1
                                    userInfo:@{NSLocalizedDescriptionKey: @"模型加载失败"}];
        }
        return NO;
    }

    llama_context_params ctxParams = llama_context_default_params();
    ctxParams.n_ctx = 2048;                // 聊天够用
    ctxParams.n_batch = 512;
    ctxParams.n_ubatch = 512;

    _ctx = llama_init_from_model(_model, ctxParams);
    if (!_ctx) {
        llama_model_free(_model);
        _model = nullptr;
        if (error) {
            *error = [NSError errorWithDomain:@"LlamaBridge"
                                        code:2
                                    userInfo:@{NSLocalizedDescriptionKey: @"context 创建失败"}];
        }
        return NO;
    }

    // sampler chain:temp 0.7, top-k 40, top-p 0.95
    llama_sampler_chain_params samplerParams = llama_sampler_chain_default_params();
    _sampler = llama_sampler_chain_init(samplerParams);
    llama_sampler_chain_add(_sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(_sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(_sampler, llama_sampler_init_top_p(0.95f, 1));

    _modelPath = [path copy];
    _loaded = YES;
    return YES;
}

- (void)generateWithPrompt:(NSString *)prompt
                maxTokens:(int32_t)maxTokens
               completion:(LlamaCompletion)completion {
    if (!_loaded || !_model || !_ctx) {
        if (completion) {
            completion(@"", NO, @"模型未加载");
        }
        return;
    }

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        std::string promptStr = std::string([prompt UTF8String]);
        std::string result;

        // 分词(新版 C API:传入 vocab 和缓冲区)
        const llama_vocab * vocab = llama_model_get_vocab(_model);
        std::vector<llama_token> tokens(512);
        int32_t nTokens = llama_tokenize(vocab, promptStr.c_str(), promptStr.size(),
                                         tokens.data(), (int32_t)tokens.size(), true, true);
        if (nTokens < 0) {
            if (completion) completion(@"", NO, @"分词失败");
            return;
        }
        tokens.resize(nTokens);

        // 一次性提交 prompt(简化原型)
        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = false;
        }
        batch.n_tokens = tokens.size();
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(_ctx, batch) != 0) {
            llama_batch_free(batch);
            if (completion) completion(@"", NO, @"prompt 解码失败");
            return;
        }
        llama_batch_free(batch);

        // 生成循环
        std::vector<llama_token> generated;
        int64_t start = llama_time_us();
        for (int32_t i = 0; i < maxTokens; i++) {
            // 采样
            llama_token newToken = llama_sampler_sample(_sampler, _ctx, -1);
            if (newToken == llama_vocab_eos(vocab)) {
                break;
            }
            generated.push_back(newToken);

            // 解码单 token
            llama_batch single = llama_batch_get_one(&newToken, 1);
            if (llama_decode(_ctx, single) != 0) {
                break;
            }
        }
        int64_t elapsed = llama_time_us() - start;
        double tokensPerSec = elapsed > 0 ? (generated.size() * 1000000.0) / (double)elapsed : 0.0;

        // 转文本(新版 API:第一个参数是 vocab,已在上面声明)
        char buf[64];
        for (llama_token t : generated) {
            int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
            if (n > 0) {
                result.append(buf, n);
            }
        }

        NSString * resultText = [NSString stringWithUTF8String:result.c_str()];
        if (!resultText) resultText = @"";
        NSString * speedNote = [NSString stringWithFormat:@"\n\n--- %.1f tok/s (%zu tokens) ---",
                                tokensPerSec, generated.size()];
        if (completion) {
            completion([resultText stringByAppendingString:speedNote], YES, nil);
        }
    });
}

- (void)unloadModel {
    if (_sampler) { llama_sampler_free(_sampler); _sampler = nullptr; }
    if (_ctx) { llama_free(_ctx); _ctx = nullptr; }
    if (_model) { llama_model_free(_model); _model = nullptr; }
    _loaded = NO;
}

@end
