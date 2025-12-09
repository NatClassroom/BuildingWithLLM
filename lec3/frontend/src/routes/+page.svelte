<script lang="ts">
  let data: any = null;
  let loading = true;
  let error: string | null = null;

  async function fetchData() {
    try {
      loading = true;
      error = null;
      const response = await fetch("http://localhost:8000/api/data");
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

  // Fetch data when component mounts
  fetchData();
</script>

<div class="container mx-auto px-4 py-8">
  <h1 class="text-3xl font-bold mb-6">API Data Display</h1>

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
