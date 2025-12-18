import OpenAI from 'openai';
import Ajv from 'ajv';
import { 
    PromptFunction, 
    LlmCommonOptions, 
    OpenRouterResponseFormat 
} from "./createLlmClient.js";
import { createLlmRetryClient, LlmRetryError, LlmRetryOptions } from "./createLlmRetryClient.js";

export class SchemaValidationError extends Error {
    constructor(message: string, options?: ErrorOptions) {
        super(message, options);
        this.name = 'SchemaValidationError';
    }
}

/**
 * Options for JSON schema prompt functions.
 * Extends common options with JSON-specific settings.
 */
export interface JsonSchemaLlmClientOptions extends LlmCommonOptions {
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
    /**
     * A custom validator function.
     * If provided, this function will be used to validate the parsed JSON data.
     * It should throw an error if the data is invalid, or return the validated data (potentially transformed).
     * If not provided, an AJV-based validator will be used.
     */
    validator?: (data: any) => any;
}

export interface CreateJsonSchemaLlmClientParams {
    prompt: PromptFunction;
    fallbackPrompt?: PromptFunction;
    disableJsonFixer?: boolean;
}

export function createJsonSchemaLlmClient(params: CreateJsonSchemaLlmClientParams) {
    const { prompt, fallbackPrompt, disableJsonFixer = false } = params;
    const llmRetryClient = createLlmRetryClient({ prompt, fallbackPrompt });
    const ajv = new Ajv({ strict: false });

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

        const { 
            maxRetries, 
            useResponseFormat: _useResponseFormat, 
            beforeValidation,
            validator,
            ...restOptions 
        } = options || {};

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
            // Only attempt to fix SyntaxErrors (JSON parsing errors).
            // Other errors (like runtime errors) should bubble up.
            if (!(parseError instanceof SyntaxError)) {
                throw parseError;
            }

            if (disableJsonFixer) {
                throw parseError;
            }

            const errorDetails = `JSON Parse Error: ${parseError.message}`;
            const fixedResponse = await _tryToFixJson(jsonDataToParse, schemaJsonString, errorDetails, options);

            if (fixedResponse) {
                try {
                    return JSON.parse(fixedResponse);
                } catch (e) {
                    throw parseError;
                }
            }

            throw parseError;
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
            // Only attempt to fix known validation errors (SchemaValidationError).
            // Arbitrary errors thrown by custom validators (e.g. "Database Error") should bubble up.
            if (!(validationError instanceof SchemaValidationError)) {
                throw validationError;
            }

            if (disableJsonFixer) {
                throw validationError;
            }

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
                    throw validationError;
                }
            }

            throw validationError;
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

        const finalMessages = [...messages];

        const systemMessageIndex = finalMessages.findIndex(m => m.role === 'system');

        if (systemMessageIndex !== -1) {
            const existingContent = finalMessages[systemMessageIndex].content;
            finalMessages[systemMessageIndex] = {
                ...finalMessages[systemMessageIndex],
                content: `${existingContent}\n${commonPromptFooter}`
            };
        } else {
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
        const defaultValidator = (data: any) => {
            try {
                const validate = ajv.compile(schema);
                const valid = validate(data);
                if (!valid) {
                    const errors = (validate.errors || []).map(e => `${e.instancePath} ${e.message}`).join(', ');
                    throw new SchemaValidationError(`AJV Validation Error: ${errors}`);
                }
                return data as T;
            } catch(error: any) {
                throw error;
            }
        };

        const validator = options?.validator ?? defaultValidator;

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
                // Only wrap SyntaxErrors (JSON parse errors) for retry.
                if (parseError instanceof SyntaxError) {
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
                // Rethrow other errors (e.g. fatal errors, runtime errors)
                throw parseError;
            }

            try {
                const validatedData = await _validateOrFix(jsonData, validator, schemaJsonString, options);
                return validatedData;
            } catch (validationError: any) {
                // Only wrap known validation errors for retry.
                if (validationError instanceof SchemaValidationError) {
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
                // Rethrow other errors
                throw validationError;
            }
        };

        const { 
            maxRetries, 
            useResponseFormat: _useResponseFormat, 
            beforeValidation,
            validator: _validator,
            ...restOptions 
        } = options || {};

        const retryOptions: LlmRetryOptions<T> & { messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] } = {
            ...restOptions,
            maxRetries,
            messages: finalMessages,
            response_format,
            validate: processResponse
        };

        return llmRetryClient.promptTextRetry(retryOptions);
    }

    return { promptJson };
}

export type JsonSchemaClient = ReturnType<typeof createJsonSchemaLlmClient>;
export type PromptJsonFunction = JsonSchemaClient['promptJson'];
