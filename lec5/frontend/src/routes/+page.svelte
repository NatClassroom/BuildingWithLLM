<script lang="ts">
  import { onMount } from "svelte";

  // Tab management
  let activeTab: "upload" | "chat" = "upload";

  // Upload tab state
  let file: File | null = null;
  let uploadedText = "";
  let chunks: any[] = [];
  let chunksWithEmbeddings: any[] = [];
  let storedDocuments: any[] = [];
  let uploadStep: "idle" | "uploading" | "chunking" | "embedding" | "storing" | "complete" = "idle";
  let uploadError: string | null = null;

  // Chat tab state
  let messages: any[] = [];
  let chatInput = "";
  let sending = false;
  let chatError: string | null = null;
  let currentPrompt: string | null = null; // Store the current full prompt
  let conversationMessages: any[] = []; // Store the conversation structure
  let retrievalQuery: string | null = null; // Store the retrieval query
  let retrievedDocuments: any[] | null = null; // Store retrieved documents
  let baseSystemInstruction: string | null = null; // Store base system instruction
  let augmentation: string | null = null; // Store augmentation details

  // All chunks from database
  let allChunks: any[] = [];
  let loadingChunks = false;

  async function handleFileSelect(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target.files && target.files[0]) {
      file = target.files[0];
      await processFile();
    }
  }

  async function processFile() {
    if (!file) return;

    uploadError = null;
    uploadStep = "uploading";

    try {
      // Step 1: Upload file and extract text
      const formData = new FormData();
      formData.append("file", file);

      const uploadResponse = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!uploadResponse.ok) {
        let errorMessage = `Upload failed: ${uploadResponse.statusText}`;
        try {
          const errorData = await uploadResponse.json();
          if (errorData.detail) {
            errorMessage = errorData.detail;
          }
        } catch {
          const errorText = await uploadResponse.text();
          if (errorText) {
            errorMessage = errorText;
          }
        }
        throw new Error(errorMessage);
      }

      const uploadData = await uploadResponse.json();
      uploadedText = uploadData.text;
      chunks = uploadData.chunks;
      uploadStep = "chunking";

      // Step 2: Generate embeddings
      await new Promise((resolve) => setTimeout(resolve, 500)); // Visual delay
      uploadStep = "embedding";

      const embedResponse = await fetch("http://localhost:8000/api/embed", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(chunks),
      });

      if (!embedResponse.ok) {
        let errorMessage = `Embedding failed: ${embedResponse.statusText}`;
        try {
          const errorData = await embedResponse.json();
          if (errorData.detail) {
            errorMessage = errorData.detail;
          }
        } catch {
          const errorText = await embedResponse.text();
          if (errorText) {
            errorMessage = errorText;
          }
        }
        throw new Error(errorMessage);
      }

      const embedData = await embedResponse.json();
      chunksWithEmbeddings = embedData.chunks;
      await new Promise((resolve) => setTimeout(resolve, 500)); // Visual delay

      // Step 3: Store in vector database
      uploadStep = "storing";

      const storeResponse = await fetch("http://localhost:8000/api/documents", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(chunksWithEmbeddings),
      });

      if (!storeResponse.ok) {
        let errorMessage = `Storage failed: ${storeResponse.statusText}`;
        try {
          const errorData = await storeResponse.json();
          if (errorData.detail) {
            errorMessage = errorData.detail;
          }
        } catch {
          const errorText = await storeResponse.text();
          if (errorText) {
            errorMessage = errorText;
          }
        }
        throw new Error(errorMessage);
      }

      storedDocuments = await storeResponse.json();
      uploadStep = "complete";
      // Refresh all chunks after storing
      await fetchAllChunks();
    } catch (e) {
      uploadError = e instanceof Error ? e.message : "Failed to process file";
      uploadStep = "idle";
    }
  }

  async function sendMessage() {
    if (!chatInput.trim() || sending) return;

    const userMessage = chatInput.trim();
    chatInput = "";
    chatError = null;

    // Add user message
    messages = [...messages, { role: "user", content: userMessage }];
    sending = true;

    try {
      // Step 1: Retrieve relevant documents
      const searchResponse = await fetch("http://localhost:8000/api/documents/search", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage,
          limit: 3,
        }),
      });

      if (!searchResponse.ok) {
        // Try to parse error response for detailed message
        let errorMessage = `Search failed: ${searchResponse.statusText}`;
        try {
          const errorData = await searchResponse.json();
          if (errorData.detail) {
            errorMessage = errorData.detail;
          }
        } catch {
          // If JSON parsing fails, use the response text
          const errorText = await searchResponse.text();
          if (errorText) {
            errorMessage = errorText;
          }
        }
        throw new Error(errorMessage);
      }

      const retrievedDocs = await searchResponse.json();

      // Store retrieval query and documents for display
      retrievalQuery = userMessage;
      retrievedDocuments = retrievedDocs;
      baseSystemInstruction = "You are a helpful assistant.";

      // Step 2: Build system instruction with context
      let systemInstruction = "You are a helpful assistant.";
      if (retrievedDocs && retrievedDocs.length > 0) {
        const context = retrievedDocs.map((doc: any) => doc.content).join("\n\n");
        systemInstruction = `You are a helpful assistant. Use the following context to answer the user's question. If the context doesn't contain relevant information, say so.

Context:
${context}`;
      }

      // Step 3: Build conversation messages (convert assistant to model, exclude system)
      const conversationMessages = messages.map((msg: any) => ({
        role: msg.role === "assistant" ? "model" : msg.role,
        content: msg.content,
      }));

      // Step 4: Send to chat endpoint with full conversation structure
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          system_instruction: systemInstruction,
          messages: conversationMessages,
          limit: 3,
        }),
      });

      if (!response.ok) {
        // Try to parse error response for detailed message
        let errorMessage = `Chat failed: ${response.statusText}`;
        try {
          const errorData = await response.json();
          if (errorData.detail) {
            errorMessage = errorData.detail;
          }
        } catch {
          // If JSON parsing fails, use the response text
          const errorText = await response.text();
          if (errorText) {
            errorMessage = errorText;
          }
        }
        
        // Create error object with status code for better handling
        const error = new Error(errorMessage) as Error & { statusCode?: number };
        error.statusCode = response.status;
        throw error;
      }

      const chatData = await response.json();
      // Add retrieved_docs to the response for display
      chatData.retrieved_docs = retrievedDocs;
      messages = [...messages, chatData];
      
      // Update the current prompt to show on the right side
      if (chatData.full_prompt) {
        currentPrompt = chatData.full_prompt;
      }

      // Fetch the conversation structure
      await fetchConversation();
    } catch (e) {
      if (e instanceof Error) {
        chatError = e.message;
        // If it's a quota error (429), we might want to show a special message
        const errorWithStatus = e as Error & { statusCode?: number };
        if (errorWithStatus.statusCode === 429) {
          // Quota error - message already contains retry info from backend
          console.warn("Quota exceeded:", e.message);
        } else {
          console.error("Chat error:", e);
        }
      } else {
        chatError = "Failed to send message. Please try again.";
      }
    } finally {
      sending = false;
    }
  }

  async function resetUpload() {
    file = null;
    uploadedText = "";
    chunks = [];
    chunksWithEmbeddings = [];
    storedDocuments = [];
    uploadStep = "idle";
    uploadError = null;
    // Refresh all chunks after reset
    await fetchAllChunks();
  }

  function formatSimilarity(similarity: number): string {
    return (similarity * 100).toFixed(1);
  }

  async function fetchAllChunks() {
    loadingChunks = true;
    try {
      const response = await fetch("http://localhost:8000/api/documents", {
        method: "GET",
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch chunks: ${response.statusText}`);
      }

      allChunks = await response.json();
    } catch (e) {
      console.error("Error fetching chunks:", e);
      allChunks = [];
    } finally {
      loadingChunks = false;
    }
  }

  async function fetchConversation() {
    // Don't fetch if there are no messages
    if (messages.length === 0) {
      conversationMessages = [];
      retrievalQuery = null;
      retrievedDocuments = null;
      baseSystemInstruction = null;
      augmentation = null;
      return;
    }

    try {
      // Build system instruction from latest retrieved docs (same logic as sendMessage)
      let systemInstruction = baseSystemInstruction || "You are a helpful assistant.";
      if (retrievedDocuments && retrievedDocuments.length > 0) {
        const context = retrievedDocuments.map((doc: any) => doc.content).join("\n\n");
        systemInstruction = `You are a helpful assistant. Use the following context to answer the user's question. If the context doesn't contain relevant information, say so.

Context:
${context}`;
      }

      // Build conversation messages (convert assistant to model, exclude system)
      const conversationMessagesToSend = messages.map((msg: any) => ({
        role: msg.role === "assistant" ? "model" : msg.role,
        content: msg.content,
      }));

      console.log("Fetching conversation with:", {
        messageCount: messages.length,
        hasRetrievedDocs: !!retrievedDocuments,
        hasSystemInstruction: !!systemInstruction,
        messages: conversationMessagesToSend.map(m => ({ role: m.role, hasContent: !!m.content })),
      });

      const response = await fetch("http://localhost:8000/api/conversation", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          system_instruction: systemInstruction,
          messages: conversationMessagesToSend,
          retrieval_query: retrievalQuery,
          retrieved_documents: retrievedDocuments,
          base_system_instruction: baseSystemInstruction,
        }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Conversation API error:", response.status, errorText);
        throw new Error(`Failed to fetch conversation: ${response.statusText}`);
      }

      const conversationData = await response.json();
      console.log("Conversation data received:", conversationData);
      console.log("Number of messages in response:", conversationData.messages?.length || 0);
      
      // Ensure we're creating a new array for reactivity
      if (conversationData.messages && Array.isArray(conversationData.messages)) {
        conversationMessages = [...conversationData.messages];
        console.log("conversationMessages updated:", conversationMessages.length, "messages");
      } else {
        console.warn("Invalid conversation data format:", conversationData);
        conversationMessages = [];
      }

      // Store augmentation details
      augmentation = conversationData.augmentation || null;
    } catch (e) {
      console.error("Error fetching conversation:", e);
      conversationMessages = [];
      augmentation = null;
    }
  }

  // Fetch all chunks on mount
  onMount(async () => {
    await fetchAllChunks();
  });
</script>

<div class="container mx-auto px-4 py-8 max-w-7xl">
  <h1 class="text-4xl font-bold text-center mb-2">RAG Demo</h1>
  <p class="text-gray-600 text-center mb-8">
    Demonstration of Retrieval-Augmented Generation (RAG) Process
  </p>

  <!-- All Chunks from Database -->
  <div class="bg-white border border-gray-200 rounded-lg p-6 shadow-sm mb-6">
    <div class="flex justify-between items-center mb-4">
      <h2 class="text-xl font-semibold">All Chunks in Database</h2>
      <button
        onclick={fetchAllChunks}
        disabled={loadingChunks}
        class="text-sm bg-gray-100 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed text-gray-700 font-medium py-1 px-3 rounded transition-colors"
      >
        {loadingChunks ? "Loading..." : "🔄 Refresh"}
      </button>
    </div>
    {#if loadingChunks}
      <div class="text-center text-gray-500 py-4">Loading chunks...</div>
    {:else if allChunks.length === 0}
      <div class="text-center text-gray-500 py-4">
        No chunks in database yet. Upload a file to get started.
      </div>
    {:else}
      <p class="text-sm text-gray-600 mb-4">
        Total: {allChunks.length} chunk{allChunks.length !== 1 ? 's' : ''} stored
      </p>
      <div class="space-y-2 max-h-64 overflow-y-auto">
        {#each allChunks as chunk}
          <div class="bg-gray-50 border border-gray-200 rounded p-3">
            <div class="flex justify-between items-center mb-1">
              <span class="text-xs font-semibold text-gray-700">Chunk ID: {chunk.id}</span>
              <span class="text-xs text-gray-500">{chunk.content.length} chars</span>
            </div>
            <p class="text-sm text-gray-800 line-clamp-2">{chunk.content}</p>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Tab Navigation -->
  <div class="flex border-b border-gray-200 mb-6">
    <button
      class="px-6 py-3 font-medium text-sm border-b-2 transition-colors {activeTab === 'upload'
        ? 'border-blue-500 text-blue-600'
        : 'border-transparent text-gray-500 hover:text-gray-700'}"
      onclick={() => (activeTab = "upload")}
    >
      📄 Upload & Process
    </button>
    <button
      class="px-6 py-3 font-medium text-sm border-b-2 transition-colors {activeTab === 'chat'
        ? 'border-blue-500 text-blue-600'
        : 'border-transparent text-gray-500 hover:text-gray-700'}"
      onclick={() => (activeTab = "chat")}
    >
      💬 Chat with RAG
    </button>
  </div>

  <!-- Upload Tab -->
  {#if activeTab === "upload"}
    <div class="space-y-6">
      <div class="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
        <h2 class="text-2xl font-semibold mb-4">Step 1: Upload Text File</h2>
        <div class="space-y-4">
          <input
            type="file"
            accept=".txt"
            onchange={handleFileSelect}
            disabled={uploadStep !== "idle" && uploadStep !== "complete"}
            class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50"
          />
          {#if uploadError}
            <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
              <div class="flex items-start gap-2">
                <span class="text-lg">⚠️</span>
                <div class="flex-1">
                  <p class="text-sm font-semibold mb-1">Error</p>
                  <p class="text-sm">{uploadError}</p>
                </div>
                <button
                  onclick={() => (uploadError = null)}
                  class="text-red-700 hover:text-red-900 text-lg font-bold"
                  aria-label="Dismiss error"
                >
                  ×
                </button>
              </div>
            </div>
          {/if}
        </div>
      </div>

      <!-- Step 2: Text Extraction -->
      {#if uploadStep !== "idle"}
        <div class="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
          <h2 class="text-2xl font-semibold mb-4">
            Step 2: Text Extraction
            {#if uploadStep === "uploading"}
              <span class="text-blue-500 text-lg">⏳ Processing...</span>
            {:else if uploadStep !== "idle"}
              <span class="text-green-500 text-lg">✓ Complete</span>
            {/if}
          </h2>
          {#if uploadedText}
            <div class="bg-gray-50 border border-gray-200 rounded p-4 max-h-48 overflow-y-auto">
              <p class="text-sm text-gray-700 whitespace-pre-wrap">{uploadedText}</p>
            </div>
          {/if}
        </div>
      {/if}

      <!-- Step 3: Chunking -->
      {#if uploadStep !== "idle" && uploadStep !== "uploading"}
        <div class="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
          <h2 class="text-2xl font-semibold mb-4">
            Step 3: Text Chunking
            {#if uploadStep === "chunking"}
              <span class="text-blue-500 text-lg">⏳ Chunking...</span>
            {:else if uploadStep !== "idle" && uploadStep !== "uploading"}
              <span class="text-green-500 text-lg">✓ Complete</span>
            {/if}
          </h2>
          {#if chunks.length > 0}
            <p class="text-sm text-gray-600 mb-4">
              Text split into {chunks.length} chunks
            </p>
            <div class="space-y-2 max-h-64 overflow-y-auto">
              {#each chunks as chunk}
                <div class="bg-blue-50 border border-blue-200 rounded p-3">
                  <div class="flex justify-between items-center mb-1">
                    <span class="text-xs font-semibold text-blue-700">Chunk {chunk.index + 1}</span>
                    <span class="text-xs text-gray-500">{chunk.text.length} chars</span>
                  </div>
                  <p class="text-sm text-gray-800 line-clamp-2">{chunk.text}</p>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      {/if}

      <!-- Step 4: Embedding -->
      {#if uploadStep !== "idle" && uploadStep !== "uploading" && uploadStep !== "chunking"}
        <div class="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
          <h2 class="text-2xl font-semibold mb-4">
            Step 4: Vector Embedding
            {#if uploadStep === "embedding"}
              <span class="text-blue-500 text-lg">⏳ Generating embeddings...</span>
            {:else if uploadStep !== "idle" && uploadStep !== "uploading" && uploadStep !== "chunking"}
              <span class="text-green-500 text-lg">✓ Complete</span>
            {/if}
          </h2>
          {#if chunksWithEmbeddings.length > 0}
            <p class="text-sm text-gray-600 mb-4">
              Generated {chunksWithEmbeddings.length} embeddings (768 dimensions each)
            </p>
            <div class="space-y-2 max-h-64 overflow-y-auto">
              {#each chunksWithEmbeddings as chunk}
                <div class="bg-purple-50 border border-purple-200 rounded p-3">
                  <div class="flex justify-between items-center mb-1">
                    <span class="text-xs font-semibold text-purple-700">Chunk {chunk.index + 1}</span>
                    <span class="text-xs text-gray-500">
                      Vector: [{chunk.embedding.slice(0, 3).map((v: number) => v.toFixed(3)).join(", ")}, ...]
                    </span>
                  </div>
                  <p class="text-sm text-gray-800 line-clamp-1">{chunk.text}</p>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      {/if}

      <!-- Step 5: Vector Database Storage -->
      {#if uploadStep !== "idle" && uploadStep !== "uploading" && uploadStep !== "chunking" && uploadStep !== "embedding"}
        <div class="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
          <h2 class="text-2xl font-semibold mb-4">
            Step 5: Vector Database Storage
            {#if uploadStep === "storing"}
              <span class="text-blue-500 text-lg">⏳ Storing...</span>
            {:else if uploadStep === "complete"}
              <span class="text-green-500 text-lg">✓ Complete</span>
            {/if}
          </h2>
          {#if storedDocuments.length > 0}
            <p class="text-sm text-gray-600 mb-4">
              Stored {storedDocuments.length} documents in vector database
            </p>
            <div class="space-y-2 max-h-64 overflow-y-auto">
              {#each storedDocuments as doc}
                <div class="bg-green-50 border border-green-200 rounded p-3">
                  <div class="flex justify-between items-center mb-1">
                    <span class="text-xs font-semibold text-green-700">Document ID: {doc.id}</span>
                  </div>
                  <p class="text-sm text-gray-800 line-clamp-1">{doc.content}</p>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      {/if}

      {#if uploadStep === "complete"}
        <div class="flex justify-center">
          <button
            onclick={resetUpload}
            class="bg-gray-500 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded"
          >
            Reset & Upload Another File
          </button>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Chat Tab -->
  {#if activeTab === "chat"}
    <div class="space-y-6">
      <div class="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
        <h2 class="text-2xl font-semibold mb-4">Chat Interface</h2>
        <p class="text-sm text-gray-600 mb-4">
          Ask questions about the uploaded documents. The system will retrieve relevant chunks and use them to generate answers.
        </p>

        <!-- Split Screen Layout -->
        <div class="flex gap-4" style="min-height: 600px;">
          <!-- Left Side: Chat Interface -->
          <div class="flex-1 flex flex-col">
            <!-- Chat Messages -->
            <div class="space-y-4 mb-4 flex-1 overflow-y-auto border border-gray-200 rounded p-4 bg-gray-50">
              {#if messages.length === 0}
                <div class="text-center text-gray-500 py-8">
                  <p>No messages yet. Start a conversation!</p>
                </div>
              {:else}
                {#each messages as message}
                  <div class="flex {message.role === 'user' ? 'justify-end' : 'justify-start'}">
                    <div
                      class="max-w-[80%] rounded-lg p-4 {message.role === 'user'
                        ? 'bg-blue-500 text-white'
                        : 'bg-white border border-gray-200 text-gray-800'}"
                    >
                      <div class="font-semibold mb-1 text-xs opacity-75">
                        {message.role === "user" ? "You" : "Assistant"}
                      </div>
                      <div class="text-sm whitespace-pre-wrap">{message.content}</div>

                      <!-- Show retrieved documents for assistant messages -->
                      {#if message.role === "assistant" && message.retrieved_docs && message.retrieved_docs.length > 0}
                        <div class="mt-3 pt-3 border-t border-gray-300">
                          <div class="text-xs font-semibold mb-2 opacity-75">
                            📚 Retrieved Context ({message.retrieved_docs.length} chunks):
                          </div>
                          <div class="space-y-2">
                            {#each message.retrieved_docs as doc}
                              <div class="bg-yellow-50 border border-yellow-200 rounded p-2 text-xs">
                                <div class="flex justify-between items-center mb-1">
                                  <span class="font-semibold text-yellow-700">
                                    Similarity: {formatSimilarity(doc.similarity)}%
                                  </span>
                                  <span class="text-gray-500">ID: {doc.id}</span>
                                </div>
                                <p class="text-gray-700 line-clamp-2">{doc.content}</p>
                              </div>
                            {/each}
                          </div>
                        </div>
                      {/if}
                    </div>
                  </div>
                {/each}
              {/if}
            </div>

            {#if chatError}
              <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
                <div class="flex items-start gap-2">
                  <span class="text-lg">⚠️</span>
                  <div class="flex-1">
                    <p class="text-sm font-semibold mb-1">Error</p>
                    <p class="text-sm">{chatError}</p>
                    {#if chatError.includes("quota") || chatError.includes("Quota")}
                      <p class="text-xs mt-2 text-red-600">
                        💡 Tip: The free tier has a daily limit. You can wait for the quota to reset or upgrade your plan.
                      </p>
                    {/if}
                  </div>
                  <button
                    onclick={() => (chatError = null)}
                    class="text-red-700 hover:text-red-900 text-lg font-bold"
                    aria-label="Dismiss error"
                  >
                    ×
                  </button>
                </div>
              </div>
            {/if}

            <!-- Chat Input -->
            <form
              onsubmit={(e) => {
                e.preventDefault();
                sendMessage();
              }}
              class="flex gap-2"
            >
              <input
                type="text"
                bind:value={chatInput}
                placeholder="Ask a question about the uploaded documents..."
                disabled={sending}
                class="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={sending || !chatInput.trim()}
                class="bg-blue-500 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-bold py-2 px-6 rounded transition-colors"
              >
                {sending ? "Sending..." : "Send"}
              </button>
            </form>
          </div>

          <!-- Right Side: Conversation Structure Display -->
          <div class="flex-1 flex flex-col border-l border-gray-300 pl-4">
            <h3 class="text-lg font-semibold mb-3">Conversation Structure</h3>
            <p class="text-xs text-gray-500 mb-3">
              Complete conversation (list of messages) sent to the LLM
            </p>
            <div class="flex-1 overflow-y-auto border border-gray-200 rounded p-4 bg-gray-50 space-y-4">
              <!-- Retrieval Query Section -->
              {#if retrievalQuery}
                <div class="bg-yellow-50 border border-yellow-200 rounded p-3">
                  <div class="text-xs font-semibold text-yellow-700 mb-2">
                    🔍 Retrieval Query:
                  </div>
                  <p class="text-sm text-gray-800">{retrievalQuery}</p>
                </div>
              {/if}

              <!-- Retrieved Chunks Section -->
              {#if retrievedDocuments && retrievedDocuments.length > 0}
                <div class="bg-orange-50 border border-orange-200 rounded p-3">
                  <div class="text-xs font-semibold text-orange-700 mb-2">
                    📚 Retrieved Chunks ({retrievedDocuments.length}):
                  </div>
                  <div class="space-y-2">
                    {#each retrievedDocuments as doc}
                      <div class="bg-white border border-orange-200 rounded p-2">
                        <div class="flex justify-between items-center mb-1">
                          <span class="text-xs font-semibold text-orange-700">
                            Similarity: {formatSimilarity(doc.similarity)}%
                          </span>
                          <span class="text-xs text-gray-500">ID: {doc.id}</span>
                        </div>
                        <p class="text-xs text-gray-700 line-clamp-3">{doc.content}</p>
                      </div>
                    {/each}
                  </div>
                </div>
              {/if}

              <!-- Augmentation Section -->
              {#if augmentation}
                <div class="bg-indigo-50 border border-indigo-200 rounded p-3">
                  <div class="text-xs font-semibold text-indigo-700 mb-2">
                    🔧 Augmentation Process:
                  </div>
                  <pre class="text-xs text-gray-700 whitespace-pre-wrap font-mono bg-white border border-indigo-200 rounded p-2 max-h-64 overflow-y-auto">{augmentation}</pre>
                </div>
              {/if}

              <!-- Conversation Messages -->
              {#if conversationMessages && conversationMessages.length > 0}
                <div class="border-t border-gray-300 pt-3">
                  <div class="text-xs font-semibold text-gray-700 mb-2">
                    💬 Conversation Messages:
                  </div>
                  <div class="space-y-3">
                    {#each conversationMessages as msg, index}
                      <div class="border-l-4 pl-3 {msg.role === 'system' ? 'border-purple-500 bg-purple-50' : msg.role === 'user' ? 'border-blue-500 bg-blue-50' : 'border-green-500 bg-green-50'} rounded p-3">
                        <div class="flex items-center gap-2 mb-2">
                          <span class="text-xs font-semibold uppercase {msg.role === 'system' ? 'text-purple-700' : msg.role === 'user' ? 'text-blue-700' : 'text-green-700'}">
                            {msg.role}
                          </span>
                          <span class="text-xs text-gray-500">({index + 1})</span>
                        </div>
                        <pre class="text-xs text-gray-700 whitespace-pre-wrap font-mono">{msg.content || '(empty)'}</pre>
                      </div>
                    {/each}
                  </div>
                </div>
              {:else}
                <div class="text-center text-gray-500 py-8">
                  <p>Submit a message to see the conversation structure</p>
                  {#if conversationMessages !== undefined}
                    <p class="text-xs mt-2">Debug: conversationMessages is {conversationMessages === null ? 'null' : 'empty array'}</p>
                  {/if}
                </div>
              {/if}
            </div>
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .line-clamp-1 {
    display: -webkit-box;
    -webkit-line-clamp: 1;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
</style>
