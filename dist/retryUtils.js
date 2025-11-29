"use strict";
// retryUtils.ts
Object.defineProperty(exports, "__esModule", { value: true });
exports.executeWithRetry = executeWithRetry;
/**
 * Executes an operation with a retry mechanism.
 *
 * @param operation - An async function representing the operation to be tried.
 *                    It receives the current attempt number and feedback from the previous validation.
 * @param validateAndProcess - An async function that processes the raw result from the operation
 *                             and validates it. It returns a RetryValidationResult.
 * @param maxRetries - The maximum number of retries after the initial attempt (e.g., 2 means 3 total attempts).
 * @param initialFeedbackForOperation - Optional initial feedback to pass to the very first call of the operation.
 * @param shouldRetryError - Optional function to determine if a specific error should trigger a retry. Returns true to retry, false to throw immediately.
 * @returns A Promise that resolves with the validated data if successful.
 * @throws An error if all attempts fail or a critical failure occurs.
 */
async function executeWithRetry(operation, validateAndProcess, maxRetries, initialFeedbackForOperation, shouldRetryError) {
    let currentFeedbackForOperation = initialFeedbackForOperation;
    for (let attemptNumber = 0; attemptNumber <= maxRetries; attemptNumber++) {
        if (attemptNumber > 0) {
            // Exponential backoff with jitter.
            const baseDelay = 1000; // 1 second
            const backoffTime = baseDelay * Math.pow(2, attemptNumber - 1);
            const jitter = backoffTime * (Math.random() * 0.2); // Add up to 20% jitter
            const totalDelay = backoffTime + jitter;
            console.log(`Retrying operation... Attempt ${attemptNumber + 1} of ${maxRetries + 1}. Waiting for ${Math.round(totalDelay)}ms.`);
            await new Promise(resolve => setTimeout(resolve, totalDelay));
        }
        let rawResult;
        try {
            rawResult = await operation(attemptNumber, currentFeedbackForOperation);
        }
        catch (opError) {
            // Check if we should stop retrying based on the error type
            if (shouldRetryError && !shouldRetryError(opError)) {
                throw opError;
            }
            // Error directly from the operation (e.g., network failure, `this.ask` throws)
            if (attemptNumber >= maxRetries) {
                // On the final attempt, throw an error that preserves the entire causal chain.
                throw new Error(`Operation failed on final attempt ${attemptNumber + 1}. See cause for details.`, { cause: opError });
            }
            // Provide feedback about the operation's own exception for the next attempt
            currentFeedbackForOperation = {
                type: 'OPERATION_EXCEPTION', // You'll need to define this in your FeedbackType
                message: opError.message,
                rawResponseSnippet: "N/A - Operation threw before producing a response.",
                // Include the cause's stack if available, otherwise the operation error's stack
                details: opError.cause?.stack || opError.stack
            }; // Cast as FeedbackType, ensure your type supports this structure
            continue; // Go to the next attempt
        }
        const validationOutcome = await validateAndProcess(rawResult, attemptNumber);
        if (validationOutcome.isValid && validationOutcome.data !== undefined) {
            return validationOutcome.data;
        }
        currentFeedbackForOperation = validationOutcome.feedbackForNextAttempt;
        if (validationOutcome.isCriticalFailure) {
            throw new Error(`Critical failure encountered on attempt ${attemptNumber + 1}. Reason: ${JSON.stringify(currentFeedbackForOperation) || 'Unknown critical failure'}`);
        }
        if (attemptNumber >= maxRetries) {
            throw new Error(`All ${maxRetries + 1} attempts failed. Last failure reason: ${JSON.stringify(currentFeedbackForOperation)}`);
        }
    }
    // This line should be theoretically unreachable if maxRetries >= 0
    throw new Error("Exited retry loop unexpectedly. This indicates an issue with the retry logic itself.");
}
