<script lang="ts">
  import { onMount } from "svelte";

  let documents: any[] = [];
  let searchResults: any[] = [];
  let loading = false;
  let searching = false;
  let error: string | null = null;
  let formError: string | null = null;
  let submitting = false;
  let content = "";
  let searchQuery = "";
  let showSearchResults = false;

  async function fetchDocuments() {
    try {
      loading = true;
      error = null;
      const response = await fetch("http://localhost:8000/api/documents");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      documents = await response.json();
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to fetch documents";
    } finally {
      loading = false;
    }
  }

  async function handleSubmit(event: Event) {
    event.preventDefault();
    formError = null;
    submitting = true;

    try {
      const response = await fetch("http://localhost:8000/api/documents", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          content: content.trim(),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `HTTP error! status: ${response.status}`
        );
      }

      // Reset form
      content = "";

      // Refresh the documents list
      await fetchDocuments();
    } catch (e) {
      formError = e instanceof Error ? e.message : "Failed to create document";
    } finally {
      submitting = false;
    }
  }

  async function handleSearch(event: Event) {
    event.preventDefault();
    if (!searchQuery.trim()) return;

    try {
      searching = true;
      error = null;
      const response = await fetch(
        "http://localhost:8000/api/documents/search",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query: searchQuery.trim(),
            limit: 5,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      searchResults = await response.json();
      showSearchResults = true;
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to search documents";
      showSearchResults = false;
    } finally {
      searching = false;
    }
  }

  function formatSimilarity(similarity: number): string {
    return (similarity * 100).toFixed(1);
  }

  onMount(() => {
    fetchDocuments();
  });
</script>

<div class="container mx-auto px-4 py-8 max-w-6xl">
  <h1 class="text-3xl font-bold mb-2">Vector Database Demo</h1>
  <p class="text-gray-600 mb-8">
    Demonstrate semantic search using vector embeddings and similarity retrieval
  </p>

  <div class="grid md:grid-cols-2 gap-6 mb-8">
    <!-- Add Document Form -->
    <div class="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
      <h2 class="text-xl font-semibold mb-4">Add Document</h2>
      <p class="text-sm text-gray-600 mb-4">
        Add a document to the vector database. It will be automatically
        embedded.
      </p>
      <form onsubmit={handleSubmit}>
        <div class="space-y-4">
          <div>
            <label
              for="content"
              class="block text-sm font-medium text-gray-700 mb-1"
            >
              Document Content
            </label>
            <textarea
              id="content"
              bind:value={content}
              required
              rows="6"
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              placeholder="Enter document content (e.g., 'Python is a high-level programming language...')"
              disabled={submitting}
            ></textarea>
          </div>
          {#if formError}
            <div
              class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded"
            >
              <p class="text-sm">{formError}</p>
            </div>
          {/if}
          <button
            type="submit"
            disabled={submitting}
            class="w-full bg-green-500 hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-bold py-2 px-4 rounded transition-colors"
          >
            {submitting ? "Adding..." : "Add Document"}
          </button>
        </div>
      </form>
    </div>

    <!-- Search Form -->
    <div class="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
      <h2 class="text-xl font-semibold mb-4">Semantic Search</h2>
      <p class="text-sm text-gray-600 mb-4">
        Search for similar documents using vector similarity (cosine
        similarity).
      </p>
      <form onsubmit={handleSearch}>
        <div class="space-y-4">
          <div>
            <label
              for="searchQuery"
              class="block text-sm font-medium text-gray-700 mb-1"
            >
              Search Query
            </label>
            <input
              type="text"
              id="searchQuery"
              bind:value={searchQuery}
              required
              class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
              placeholder="Enter your search query..."
              disabled={searching}
            />
          </div>
          <button
            type="submit"
            disabled={searching || !searchQuery.trim()}
            class="w-full bg-blue-500 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-bold py-2 px-4 rounded transition-colors"
          >
            {searching ? "Searching..." : "Search"}
          </button>
        </div>
      </form>
    </div>
  </div>

  {#if error}
    <div
      class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4"
    >
      <p class="font-bold">Error:</p>
      <p>{error}</p>
    </div>
  {/if}

  <!-- Search Results -->
  {#if showSearchResults && searchResults.length > 0}
    <div class="bg-white border border-gray-200 rounded-lg p-6 shadow-sm mb-6">
      <h2 class="text-xl font-semibold mb-4">
        Search Results ({searchResults.length})
      </h2>
      <div class="space-y-4">
        {#each searchResults as result}
          <div
            class="bg-blue-50 border border-blue-200 rounded-lg p-4 hover:shadow-md transition-shadow"
          >
            <div class="flex justify-between items-start mb-2">
              <span class="text-xs font-semibold text-blue-700">
                Similarity: {formatSimilarity(result.similarity)}%
              </span>
              <span class="text-xs text-gray-500">ID: {result.id}</span>
            </div>
            <p class="text-gray-800">{result.content}</p>
          </div>
        {/each}
      </div>
      <button
        onclick={() => {
          showSearchResults = false;
          searchQuery = "";
        }}
        class="mt-4 text-sm text-gray-600 hover:text-gray-800"
      >
        Clear Results
      </button>
    </div>
  {:else if showSearchResults && searchResults.length === 0}
    <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
      <p class="text-yellow-800">No similar documents found.</p>
    </div>
  {/if}

  <!-- All Documents -->
  <div class="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
    <div class="flex justify-between items-center mb-4">
      <h2 class="text-xl font-semibold">All Documents ({documents.length})</h2>
      <button
        onclick={fetchDocuments}
        disabled={loading}
        class="text-sm bg-gray-100 hover:bg-gray-200 disabled:bg-gray-100 disabled:cursor-not-allowed text-gray-700 font-medium py-2 px-4 rounded transition-colors"
      >
        {loading ? "Loading..." : "Refresh"}
      </button>
    </div>

    {#if loading}
      <div class="text-center py-8">
        <p class="text-gray-600">Loading documents...</p>
      </div>
    {:else if documents.length === 0}
      <div class="text-center py-8">
        <p class="text-gray-600">
          No documents yet. Add your first document above!
        </p>
      </div>
    {:else}
      <div class="space-y-3">
        {#each documents as doc}
          <div
            class="bg-gray-50 border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
          >
            <div class="flex justify-between items-start mb-2">
              <span class="text-xs font-semibold text-gray-500"
                >Document ID: {doc.id}</span
              >
            </div>
            <p class="text-gray-800">{doc.content}</p>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>
