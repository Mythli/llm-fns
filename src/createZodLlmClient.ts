import OpenAI from 'openai'; // Import the OpenAI library
import * as z from "zod";
import { PromptFunction, LlmPromptOptions, OpenRouterResponseFormat, IsPromptCachedFunction } from "./createLlmClient.js";
import { createLlmRetryClient, LlmRetryError, LlmRetryOptions } from "./createLlmRetryClient.js";
import { ZodError, ZodTypeAny } from "zod";

export type ZodLlmClientOptions = Omit<LlmPromptOptions, 'messages' | 'response_format'> & {
    maxRetries?: number;
    /**
     * If true, passes `response_format: { type: 'json_object' }` to the model.
     * If false, only includes the schema in the system prompt.
     * Defaults to true.
     */
    useResponseFormat?: boolean;
    /**
     * A hook to process the parsed JSON data before it is validated against the Zod schema.
     * This can be used to merge partial results or perform other transformations.
     * @param data The parsed JSON data from the LLM response.
     * @returns The processed data to be validated.
     */
    beforeValidation?: (data: any) => any;
}

export interface CreateZodLlmClientParams {
    prompt: PromptFunction;
    isPromptCached: IsPromptCachedFunction;
    fallbackPrompt?: PromptFunction;
    disableJsonFixer?: boolean;
}

function isZodSchema(obj: any): obj is ZodTypeAny {
    return (
        typeof obj === 'object' &&
        obj !== null &&
        'parse' in obj &&
        '_def' in obj
    );
}

export interface NormalizedZodArgs<T extends ZodTypeAny> {
    mainInstruction: string;
    userMessagePayload: string | OpenAI.Chat.Completions.ChatCompletionContentPart[];
    dataExtractionSchema: T;
    options?: ZodLlmClientOptions;
}

export function normalizeZodArgs<T extends ZodTypeAny>(
    arg1: string | T,
    arg2?: string | OpenAI.Chat.Completions.ChatCompletionContentPart[] | T | ZodLlmClientOptions,
    arg3?: T | ZodLlmClientOptions,
    arg4?: ZodLlmClientOptions
): NormalizedZodArgs<T> {
    if (isZodSchema(arg1)) {
        // Case 1: promptZod(schema, options?)
        return {
            mainInstruction: "Generate a valid JSON object based on the schema.",
            userMessagePayload: "Generate the data.",
            dataExtractionSchema: arg1,
            options: arg2 as ZodLlmClientOptions | undefined
        };
    }

    if (typeof arg1 === 'string') {
        if (isZodSchema(arg2)) {
            // Case 2: promptZod(prompt, schema, options?)
            return {
                mainInstruction: "You are a helpful assistant that outputs JSON matching the provided schema.",
                userMessagePayload: arg1,
                dataExtractionSchema: arg2 as T,
                options: arg3 as ZodLlmClientOptions | undefined
            };
        }

        // Case 3: promptZod(system, user, schema, options?)
        return {
            mainInstruction: arg1,
            userMessagePayload: arg2 as string | OpenAI.Chat.Completions.ChatCompletionContentPart[],
            dataExtractionSchema: arg3 as T,
            options: arg4
        };
    }

    throw new Error("Invalid arguments passed to promptZod");
}

export function createZodLlmClient(params: CreateZodLlmClientParams) {
    const { prompt, isPromptCached, fallbackPrompt, disableJsonFixer = false } = params;
    const llmRetryClient = createLlmRetryClient({ prompt, fallbackPrompt });

    async function _tryToFixJson(
        brokenResponse: string,
        schemaJsonString: string,
        errorDetails: string,
        options?: ZodLlmClientOptions
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
        options: ZodLlmClientOptions | undefined
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

    async function _validateOrFixSchema<SchemaType extends ZodTypeAny>(
        jsonData: any,
        dataExtractionSchema: SchemaType,
        schemaJsonString: string,
        options: ZodLlmClientOptions | undefined
    ): Promise<z.infer<SchemaType>> {
        try {
            if (options?.beforeValidation) {
                jsonData = options.beforeValidation(jsonData);
            }
            return dataExtractionSchema.parse(jsonData);
        } catch (validationError: any) {
            if (!(validationError instanceof ZodError) || disableJsonFixer) {
                throw validationError;
            }

            // Attempt a one-time fix for schema validation errors.
            const errorDetails = `Schema Validation Error: ${JSON.stringify(validationError.format(), null, 2)}`;
            const fixedResponse = await _tryToFixJson(JSON.stringify(jsonData, null, 2), schemaJsonString, errorDetails, options);

            if (fixedResponse) {
                try {
                    let fixedJsonData = JSON.parse(fixedResponse);
                    if (options?.beforeValidation) {
                        fixedJsonData = options.beforeValidation(fixedJsonData);
                    }
                    return dataExtractionSchema.parse(fixedJsonData);
                } catch (e) {
                    // Fix-up failed, throw original validation error
                    throw validationError;
                }
            }

            throw validationError; // if no fixed response
        }
    }

    function _getZodPromptConfig<T extends ZodTypeAny>(
        mainInstruction: string,
        dataExtractionSchema: T,
        options?: ZodLlmClientOptions
    ) {
        const schema = z.toJSONSchema(dataExtractionSchema, {
            unrepresentable: 'any'
        })
        const schemaJsonString = JSON.stringify(schema);

        const commonPromptFooter = `
Your response MUST be a single JSON entity (object or array) that strictly adheres to the following JSON schema.
Do NOT include any other text, explanations, or markdown formatting (like \`\`\`json) before or after the JSON entity.

JSON schema:
${schemaJsonString}`;

        const finalMainInstruction = `${mainInstruction}\n${commonPromptFooter}`;

        const useResponseFormat = options?.useResponseFormat ?? true;
        const response_format: OpenRouterResponseFormat | undefined = useResponseFormat
            ? { type: 'json_object' }
            : undefined;

        return { finalMainInstruction, schemaJsonString, response_format };
    }

    async function promptZod<T extends ZodTypeAny>(
        schema: T,
        options?: ZodLlmClientOptions
    ): Promise<z.infer<T>>;
    async function promptZod<T extends ZodTypeAny>(
        prompt: string,
        schema: T,
        options?: ZodLlmClientOptions
    ): Promise<z.infer<T>>;
    async function promptZod<T extends ZodTypeAny>(
        mainInstruction: string,
        userMessagePayload: string | OpenAI.Chat.Completions.ChatCompletionContentPart[],
        dataExtractionSchema: T,
        options?: ZodLlmClientOptions
    ): Promise<z.infer<T>>;
    async function promptZod<T extends ZodTypeAny>(
        arg1: string | T,
        arg2?: string | OpenAI.Chat.Completions.ChatCompletionContentPart[] | T | ZodLlmClientOptions,
        arg3?: T | ZodLlmClientOptions,
        arg4?: ZodLlmClientOptions
    ): Promise<z.infer<T>> {
        const { mainInstruction, userMessagePayload, dataExtractionSchema, options } = normalizeZodArgs(arg1, arg2, arg3, arg4);

        const { finalMainInstruction, schemaJsonString, response_format } = _getZodPromptConfig(
            mainInstruction,
            dataExtractionSchema,
            options
        );

        const retryOptions: LlmRetryOptions = {
            ...options,
            response_format,
        };

        const processResponse = async (llmResponseString: string): Promise<z.infer<T>> => {
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
                const validatedData = await _validateOrFixSchema(jsonData, dataExtractionSchema, schemaJsonString, options);
                return validatedData;
            } catch (validationError: any) {
                if (validationError instanceof ZodError) {
                    const rawResponseForError = JSON.stringify(jsonData, null, 2);
                    const errorDetails = JSON.stringify(validationError.format(), null, 2);
                    const errorMessage = `Your previous response resulted in an error.
Error Type: SCHEMA_VALIDATION_ERROR
Error Details: ${errorDetails}
The response was valid JSON but did not conform to the required schema. Please review the errors and the schema to provide a corrected response.`;
                    throw new LlmRetryError(
                        errorMessage,
                        'CUSTOM_ERROR',
                        validationError.format(),
                        rawResponseForError
                    );
                }
                // For other errors, rethrow and let LlmRetryClient handle as critical.
                throw validationError;
            }
        };

        // Use promptTextRetry because we expect a string response to parse as JSON
        return llmRetryClient.promptTextRetry(
            finalMainInstruction,
            userMessagePayload,
            processResponse,
            retryOptions
        );
    }

    async function isPromptZodCached<T extends ZodTypeAny>(
        schema: T,
        options?: ZodLlmClientOptions
    ): Promise<boolean>;
    async function isPromptZodCached<T extends ZodTypeAny>(
        prompt: string,
        schema: T,
        options?: ZodLlmClientOptions
    ): Promise<boolean>;
    async function isPromptZodCached<T extends ZodTypeAny>(
        mainInstruction: string,
        userMessagePayload: string | OpenAI.Chat.Completions.ChatCompletionContentPart[],
        dataExtractionSchema: T,
        options?: ZodLlmClientOptions
    ): Promise<boolean>;
    async function isPromptZodCached<T extends ZodTypeAny>(
        arg1: string | T,
        arg2?: string | OpenAI.Chat.Completions.ChatCompletionContentPart[] | T | ZodLlmClientOptions,
        arg3?: T | ZodLlmClientOptions,
        arg4?: ZodLlmClientOptions
    ): Promise<boolean> {
        const { mainInstruction, userMessagePayload, dataExtractionSchema, options } = normalizeZodArgs(arg1, arg2, arg3, arg4);

        const { finalMainInstruction, response_format } = _getZodPromptConfig(
            mainInstruction,
            dataExtractionSchema,
            options
        );

        const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
            { role: "system", content: finalMainInstruction },
            { role: "user", content: userMessagePayload }
        ];

        const { maxRetries, useResponseFormat: _u, beforeValidation, ...restOptions } = options || {};

        return isPromptCached({
            messages,
            response_format,
            ...restOptions
        });
    }

    return { promptZod, isPromptZodCached };
}
