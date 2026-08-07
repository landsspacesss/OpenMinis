//
//  ContentView.swift
//  原型验证 UI:选择 GGUF 模型文件 → 推理 → 显示速度
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var modelPath: String = ""
    @State private var prompt: String = "用中文介绍一下你自己"
    @State private var output: String = "未加载模型"
    @State private var isGenerating = false
    @State private var isShowingPicker = false
    private let bridge = LlamaBridge()

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                // 模型选择
                Button {
                    isShowingPicker = true
                } label: {
                    Label(modelPath.isEmpty ? "选择 GGUF 模型文件" : (modelPath as NSString).lastPathComponent,
                          systemImage: "folder")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .padding(.horizontal)

                // 提示词输入
                TextField("输入提示词", text: $prompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .padding(.horizontal)

                // 生成按钮
                Button {
                    generate()
                } label: {
                    Text(isGenerating ? "生成中..." : "生成")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isGenerating || modelPath.isEmpty)
                .padding(.horizontal)

                // 输出
                ScrollView {
                    Text(output)
                        .font(.body.monospaced())
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(.secondarySystemBackground))
                .cornerRadius(12)
                .padding(.horizontal)
            }
            .navigationTitle("llama.cpp 原型验证")
            .fileImporter(
                isPresented: $isShowingPicker,
                allowedContentTypes: [.data],
                allowsMultipleSelection: false
            ) { result in
                if case .success(let urls) = result, let url = urls.first {
                    // 获取安全访问权限
                    let didAccess = url.startAccessingSecurityScopedResource()
                    modelPath = url.path
                    output = "模型已选择,点击生成开始推理\n(文件访问: \(didAccess ? "OK" : "FAILED"))"
                }
            }
        }
    }

    private func generate() {
        guard !modelPath.isEmpty else { return }
        isGenerating = true
        output = "加载模型并推理中...(首次加载约 10-30 秒)"
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            do {
                try bridge.loadModel(atPath: modelPath)
                bridge.generate(withPrompt: prompt, maxTokens: 64) { text, success, error in
                    DispatchQueue.main.async {
                        output = success ? text : "推理失败: \(error ?? "未知错误")"
                        isGenerating = false
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    output = "模型加载失败: \(error.localizedDescription)"
                    isGenerating = false
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
