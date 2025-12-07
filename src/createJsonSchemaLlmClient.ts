import OpenAI from 'openai';
import Ajv from 'ajv';
import { PromptFunction, LlmPromptOptions, OpenRouterResponseFormat, IsPromptCachedFunction } from "./createLlmClient.js";
import { createLlmRetryClient, LlmRetryError, LlmRetryOptions } from "./createLlmRetryClient.js";

export type JsonSchemaLlmClientOptions = Omit<LlmPromptOptions, 'messages' | 'response_format'> & {
    maxRetries?: number;
    /**
     * If true, passes `response_format: { type: 'json_object' }` to the model.
     * If false, only includes the schema in the system prompt.
     * Defaults to true.
     */
    useResponseFormat?: boolean;
    /**
     * A hook to process the parsed JSON data before it is validated.
     * This can be used to merge partial results or perform other transformations.
     * @param data The parsed JSON data from the LLM response.
     * @returns The processed data to be validated.
     */
    beforeValidation?: (data: any) => any;
}

export interface CreateJsonSchemaLlmClientParams {
    prompt: PromptFunction;
    isPromptCached: IsPromptCachedFunction;
    fallbackPrompt?: PromptFunction;
    disableJsonFixer?: boolean;
}

export function createJsonSchemaLlmClient(params: CreateJsonSchemaLlmClientParams) {
    const { prompt, isPromptCached, fallbackPrompt, disableJsonFixer = false } = params;
    const llmRetryClient = createLlmRetryClient({ prompt, fallbackPrompt });
    const ajv = new Ajv({ strict: false }); // Initialize AJV

    async function _tryToFixJson(
        brokenResponse: string,
        schemaJsonString: string,
        errorDetails: string,
        options?: JsonSchemaLlmClientOptions
    ): Promise<string | null> {
        const fixupPrompt = `
An attempt to generate a JSON object resulted in the following output, which is either not valid JSON or does not conform to the required schema.

Your task is to act as a JSON fixer. Analyze the provided "BROKEN RESPONSE" and correct it to match the "REQUIRED JSON SCHEMA".

- If the broken response contains all the necessary information to create a valid JSON object according to the schema, please provide the corrected, valid JSON object.
- If the broken response is missing essential information, or is too garbled to be fixed, please respond with the exact string: "CANNOT_FIX".
- Your response must be ONLY the corrected JSON object or the string "CANNOT_FIX". Do not include any other text, explanations, or markdown formatting.

REQUIRED JSON SCHEMA:
${schemaJsonString}

ERROR DETAILS:
${errorDetails}

BROKEN RESPONSE:
${brokenResponse}
`;

        const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
            { role: 'system', content: 'You are an expert at fixing malformed JSON data to match a specific schema.' },
            { role: 'user', content: fixupPrompt }
        ];

        const useResponseFormat = options?.useResponseFormat ?? true;
        const response_format: OpenRouterResponseFormat | undefined = useResponseFormat
            ? { type: 'json_object' }
            : undefined;

        const { maxRetries, useResponseFormat: _useResponseFormat, ...restOptions } = options || {};

        const completion = await prompt({
            messages,
            response_format,
            ...restOptions
        });

        const fixedResponse = completion.choices[0]?.message?.content;

        if (fixedResponse && fixedResponse.trim() === 'CANNOT_FIX') {
            return null;
        }

        return fixedResponse || null;
    }


    async function _parseOrFixJson(
        llmResponseString: string,
        schemaJsonString: string,
        options: JsonSchemaLlmClientOptions | undefined
    ): Promise<any> {
        let jsonDataToParse: string = llmResponseString.trim();

        // Robust handling for responses wrapped in markdown code blocks
        const codeBlockRegex = /```(?:json)?\s*([\s\S]*?)\s*```/;
        const match = codeBlockRegex.exec(jsonDataToParse);
        if (match && match[1]) {
            jsonDataToParse = match[1].trim();
        }

        if (jsonDataToParse === "") {
            throw new Error("LLM returned an empty string.");
        }

        try {
            return JSON.parse(jsonDataToParse);
        } catch (parseError: any) {
            if (disableJsonFixer) {
                throw parseError; // re-throw original error
            }

            // Attempt a one-time fix before failing.
            const errorDetails = `JSON Parse Error: ${parseError.message}`;
            const fixedResponse = await _tryToFixJson(jsonDataToParse, schemaJsonString, errorDetails, options);

            if (fixedResponse) {
                try {
                    return JSON.parse(fixedResponse);
                } catch (e) {
                    // Fix-up failed, throw original error.
                    throw parseError;
                }
            }

            throw parseError; // if no fixed response
        }
    }

    async function _validateOrFix<T>(
        jsonData: any,
        validator: (data: any) => T,
        schemaJsonString: string,
        options: JsonSchemaLlmClientOptions | undefined
    ): Promise<T> {
        try {
            if (options?.beforeValidation) {
                jsonData = options.beforeValidation(jsonData);
            }
            return validator(jsonData);
        } catch (validationError: any) {
            if (disableJsonFixer) {
                throw validationError;
            }

            // Attempt a one-time fix for schema validation errors.
            const errorDetails = `Schema Validation Error: ${validationError.message}`;
            const fixedResponse = await _tryToFixJson(JSON.stringify(jsonData, null, 2), schemaJsonString, errorDetails, options);

            if (fixedResponse) {
                try {
                    let fixedJsonData = JSON.parse(fixedResponse);
                    if (options?.beforeValidation) {
                        fixedJsonData = options.beforeValidation(fixedJsonData);
                    }
                    return validator(fixedJsonData);
                } catch (e) {
                    // Fix-up failed, throw original validation error
                    throw validationError;
                }
            }

            throw validationError; // if no fixed response
        }
    }

    function _getJsonPromptConfig(
        messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
        schema: Record<string, any>,
        options?: JsonSchemaLlmClientOptions
    ) {
        const schemaJsonString = JSON.stringify(schema);

        const commonPromptFooter = `
Your response MUST be a single JSON entity (object or array) that strictly adheres to the following JSON schema.
Do NOT include any other text, explanations, or markdown formatting (like \`\`\`json) before or after the JSON entity.

JSON schema:
${schemaJsonString}`;

        // Clone messages to avoid mutating the input
        const finalMessages = [...messages];

        // Find the first system message to append instructions to
        const systemMessageIndex = finalMessages.findIndex(m => m.role === 'system');

        if (systemMessageIndex !== -1) {
            // Append to existing system message
            const existingContent = finalMessages[systemMessageIndex].content;
            finalMessages[systemMessageIndex] = {
                ...finalMessages[systemMessageIndex],
                content: `${existingContent}\n${commonPromptFooter}`
            };
        } else {
            // Prepend new system message
            finalMessages.unshift({
                role: 'system',
                content: commonPromptFooter
            });
        }

        const useResponseFormat = options?.useResponseFormat ?? true;
        const response_format: OpenRouterResponseFormat | undefined = useResponseFormat
            ? { type: 'json_object' }
            : undefined;

        return { finalMessages, schemaJsonString, response_format };
    }

    async function promptJson<T>(
        messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
        schema: Record<string, any>,
        options?: JsonSchemaLlmClientOptions
    ): Promise<T> {
        // Always validate against the schema using AJV
        const ajvValidator = (data: any) => {
            try {
                const validate = ajv.compile(schema);
                const valid = validate(data);
                if (!valid) {
                    const errors = validate.errors?.map(e => `${e.instancePath} ${e.message}`).join(', ');
                    throw new Error(`AJV Validation Error: ${errors}`);
                }
                return data as T;
            } catch(error: any) {
                throw error;
            }
        };

        const { finalMessages, schemaJsonString, response_format } = _getJsonPromptConfig(
            messages,
            schema,
            options
        );

        const processResponse = async (llmResponseString: string): Promise<T> => {
            let jsonData: any;
            try {
                jsonData = await _parseOrFixJson(llmResponseString, schemaJsonString, options);
            } catch (parseError: any) {
                const errorMessage = `Your previous response resulted in an error.
Error Type: JSON_PARSE_ERROR
Error Details: ${parseError.message}
The response provided was not valid JSON. Please correct it.`;
                throw new LlmRetryError(
                    errorMessage,
                    'JSON_PARSE_ERROR',
                    undefined,
                    llmResponseString
                );
            }

            try {
                const validatedData = await _validateOrFix(jsonData, ajvValidator, schemaJsonString, options);
                return validatedData;
            } catch (validationError: any) {
                // We assume the validator throws an error with a meaningful message
                const rawResponseForError = JSON.stringify(jsonData, null, 2);
                const errorDetails = validationError.message;
                const errorMessage = `Your previous response resulted in an error.
Error Type: SCHEMA_VALIDATION_ERROR
Error Details: ${errorDetails}
The response was valid JSON but did not conform to the required schema. Please review the errors and the schema to provide a corrected response.`;
                throw new LlmRetryError(
                    errorMessage,
                    'CUSTOM_ERROR',
                    validationError,
                    rawResponseForError
                );
            }
        };

        const retryOptions: LlmRetryOptions<T> = {
            ...options,
            messages: finalMessages,
            response_format,
            validate: processResponse
        };

        return llmRetryClient.promptTextRetry(retryOptions);
    }

    async function isPromptJsonCached(
        messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
        schema: Record<string, any>,
        options?: JsonSchemaLlmClientOptions
    ): Promise<boolean> {
        const { finalMessages, response_format } = _getJsonPromptConfig(
            messages,
            schema,
            options
        );

        const { maxRetries, useResponseFormat: _u, beforeValidation, ...restOptions } = options || {};

        return isPromptCached({
            messages: finalMessages,
            response_format,
            ...restOptions
        });
    }

    return { promptJson, isPromptJsonCached };
}

export type JsonSchemaClient = ReturnType<typeof createJsonSchemaLlmClient>;
export type PromptJsonFunction = JsonSchemaClient['promptJson'];
