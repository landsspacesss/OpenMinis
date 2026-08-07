//
//  LlamaPrototypeApp.swift
//  原型验证 App 入口
//

import SwiftUI

@main
struct LlamaPrototypeApp: App {
    init() {
        LlamaBridge.initializeBackend()
    }
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
