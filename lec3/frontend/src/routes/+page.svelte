<script lang="ts">
  let data: any = null;
  let loading = true;
  let error: string | null = null;
  let formError: string | null = null;
  let submitting = false;
  let name = "";
  let description = "";
  import { onMount } from "svelte";

  async function fetchData() {
    try {
      loading = true;
      error = null;
      const response = await fetch("http://localhost:8000/api/items");
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      data = await response.json();
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to fetch data";
    } finally {
      loading = false;
    }
  }

  async function handleSubmit(event: Event) {
    event.preventDefault();
    formError = null;
    submitting = true;

    try {
      const response = await fetch("http://localhost:8000/api/items", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: name.trim(),
          description: description.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Reset form
      name = "";
      description = "";

      // Refresh the data to show the new item
      await fetchData();
    } catch (e) {
      formError = e instanceof Error ? e.message : "Failed to create item";
    } finally {
      submitting = false;
    }
  }

  onMount(() => {
    fetchData();
  });
</script>

<div class="container mx-auto px-4 py-8">
  <h1 class="text-3xl font-bold mb-6">API Data Display</h1>

  <!-- Add Item Form -->
  <div class="bg-white border border-gray-200 rounded-lg p-6 shadow-sm mb-6">
    <h2 class="text-xl font-semibold mb-4">Add New Item</h2>
    <form onsubmit={handleSubmit}>
      <div class="space-y-4">
        <div>
          <label
            for="name"
            class="block text-sm font-medium text-gray-700 mb-1"
          >
            Name
          </label>
          <input
            type="text"
            id="name"
            bind:value={name}
            required
            class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            placeholder="Enter item name"
            disabled={submitting}
          />
        </div>
        <div>
          <label
            for="description"
            class="block text-sm font-medium text-gray-700 mb-1"
          >
            Description
          </label>
          <textarea
            id="description"
            bind:value={description}
            required
            rows="3"
            class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            placeholder="Enter item description"
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
          {submitting ? "Adding..." : "Add Item"}
        </button>
      </div>
    </form>
  </div>

  {#if loading}
    <div class="text-center py-8">
      <p class="text-gray-600">Loading data...</p>
    </div>
  {:else if error}
    <div
      class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4"
    >
      <p class="font-bold">Error:</p>
      <p>{error}</p>
      <button
        onclick={fetchData}
        class="mt-2 bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded"
      >
        Retry
      </button>
    </div>
  {:else if data}
    <div class="space-y-4">
      <div class="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <h2 class="text-xl font-semibold mb-2">{data.message}</h2>
        <p class="text-gray-600">Total items: {data.count}</p>
      </div>

      <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {#each data.items as item}
          <div
            class="bg-white border border-gray-200 rounded-lg p-4 shadow-sm hover:shadow-md transition-shadow"
          >
            <h3 class="text-lg font-semibold mb-2">ID: {item.id}</h3>
            <p class="text-gray-800 font-medium mb-1">{item.name}</p>
            <p class="text-gray-600 text-sm">{item.description}</p>
          </div>
        {/each}
      </div>

      <button
        onclick={fetchData}
        class="mt-4 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
      >
        Refresh Data
      </button>
    </div>
  {/if}
</div>
