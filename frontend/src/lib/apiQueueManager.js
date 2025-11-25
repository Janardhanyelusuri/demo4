class ApiQueueManager {
  constructor() {
    this.queue = [];
    this.isProcessing = false;
  }

  async enqueue(apiCall, apiName) {
    console.log(`Enqueuing API call: ${apiName}`);
    return new Promise((resolve, reject) => {
      this.queue.push({
        apiCall,
        resolve,
        reject,
        apiName,
      });

      console.log(`Current queue:, this.queue.map(item => item.apiName)`);

      if (!this.isProcessing) {
        this.processQueue();
      }
    });
  }

  async processQueue() {
    if (this.isProcessing || this.queue.length === 0) {
      return;
    }

    this.isProcessing = true;
    const { apiCall, resolve, reject, apiName } = this.queue.shift();

    console.log(`Processing API call: ${apiName}`);
    console.log(`Queue length after dequeue: ${this.queue.length}`);

    try {
      const result = await apiCall();
      console.log(`Successfully processed API call: ${apiName}`);
      resolve(result);
    } catch (error) {
      console.error(`Error in API call ${apiName}:, error`);
      reject(error);
    } finally {
      this.isProcessing = false;
      if (this.queue.length > 0) {
        setTimeout(() => this.processQueue(), 500); // Add small delay between calls
      }
    }
  }
}

const apiQueue = new ApiQueueManager();
export default apiQueue;