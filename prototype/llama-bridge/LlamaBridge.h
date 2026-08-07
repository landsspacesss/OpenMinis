//
//  LlamaBridge.h
//  原型验证:Objective-C++ 桥接层,封装 llama.cpp 推理
//
//  Swift 无法直接 import llama.h(C++ 头),通过此 .mm 桥接
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// 推理结果回调
typedef void (^LlamaCompletion)(NSString * _Nonnull text, BOOL success, NSString * _Nullable error);

/// 模型信息
@interface LlamaModelInfo : NSObject
@property (nonatomic, readonly) NSString * name;
@property (nonatomic, readonly) int32_t nParams;      // 参数量(M)
@property (nonatomic, readonly) int32_t contextSize;  // context
@property (nonatomic, readonly) BOOL metalEnabled;    // Metal 后端是否启用
@end

@interface LlamaBridge : NSObject

/// 初始化后端(必须在首次使用前调用)
+ (void)initializeBackend;

/// 加载模型
- (BOOL)loadModelAtPath:(NSString *)path error:(NSError **)error;

/// 同步生成(阻塞调用,在后台线程执行)
- (void)generateWithPrompt:(NSString *)prompt
                maxTokens:(int32_t)maxTokens
               completion:(LlamaCompletion)completion;

/// 释放模型
- (void)unloadModel;

@end

NS_ASSUME_NONNULL_END
