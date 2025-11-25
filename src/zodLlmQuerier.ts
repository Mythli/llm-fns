import OpenAI from 'openai'; // Import the OpenAI library
import * as z from "zod";
import {AskGptFunction, GptAskOptions, OpenRouterResponseFormat} from "./createCachedGptAsk.js";
import { LlmReQuerier, LlmQuerierError, LlmReQuerierOptions } from "./llmReQuerier.js";
import {ZodError, ZodTypeAny} from "zod";

// LlmContentPart will now directly use OpenAI's type
// export interface LlmContentPart { ... } // This is no longer needed

// Define the structure of feedback passed between retry attempts
// This is now handled by LlmReQuerier

export type ZodLlmQuerierOptions = Omit<GptAskOptions, 'messages' | 'response_format'> & {
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

export class ZodLlmQuerier {
    protected llmReQuerier: LlmReQuerier;
    protected disableJsonFixer: boolean;

    constructor(
        protected ask: AskGptFunction,
        options: { fallbackAsk?: AskGptFunction, disableJsonFixer?: boolean } = {}
    ) {
        this.llmReQuerier = new LlmReQuerier(ask, { fallbackAsk: options.fallbackAsk });
        this.disableJsonFixer = options.disableJsonFixer ?? false;
    }

    private async _tryToFixJson(
        brokenResponse: string,
        schemaJsonString: string,
        errorDetails: string,
        options?: ZodLlmQuerierOptions
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

        const fixedResponse = await this.ask({
            messages,
            response_format,
            ...restOptions
        });

        if (fixedResponse && fixedResponse.trim() === 'CANNOT_FIX') {
            return null;
        }

        return fixedResponse;
    }


    private async _parseOrFixJson(
        llmResponseString: string,
        schemaJsonString: string,
        options: ZodLlmQuerierOptions | undefined
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
            if (this.disableJsonFixer) {
                throw parseError; // re-throw original error
            }

            // Attempt a one-time fix before failing.
            const errorDetails = `JSON Parse Error: ${parseError.message}`;
            const fixedResponse = await this._tryToFixJson(jsonDataToParse, schemaJsonString, errorDetails, options);

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

    private async _validateOrFixSchema<SchemaType extends ZodTypeAny>(
        jsonData: any,
        dataExtractionSchema: SchemaType,
        schemaJsonString: string,
        options: ZodLlmQuerierOptions | undefined
    ): Promise<z.infer<SchemaType>> {
        try {
            if (options?.beforeValidation) {
                jsonData = options.beforeValidation(jsonData);
            }
            return dataExtractionSchema.parse(jsonData);
        } catch (validationError: any) {
            if (!(validationError instanceof ZodError) || this.disableJsonFixer) {
                throw validationError;
            }

            // Attempt a one-time fix for schema validation errors.
            const errorDetails = `Schema Validation Error: ${JSON.stringify(validationError.format(), null, 2)}`;
            const fixedResponse = await this._tryToFixJson(JSON.stringify(jsonData, null, 2), schemaJsonString, errorDetails, options);

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

    public async query<T extends ZodTypeAny>(
        mainInstruction: string,
        userMessagePayload: OpenAI.Chat.Completions.ChatCompletionContentPart[], // Use OpenAI's type
        dataExtractionSchema: T,
        options?: ZodLlmQuerierOptions
    ): Promise<z.infer<T>> {
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

        const querierOptions: LlmReQuerierOptions = {
            ...options,
            response_format,
        };

        const processResponse = async (llmResponseString: string): Promise<z.infer<T>> => {
            let jsonData: any;
            try {
                jsonData = await this._parseOrFixJson(llmResponseString, schemaJsonString, options);
            } catch (parseError: any) {
                const errorMessage = `Your previous response resulted in an error.
Error Type: JSON_PARSE_ERROR
Error Details: ${parseError.message}
The response provided was not valid JSON. Please correct it.`;
                throw new LlmQuerierError(
                    errorMessage,
                    'JSON_PARSE_ERROR',
                    undefined,
                    llmResponseString
                );
            }

            try {
                const validatedData = await this._validateOrFixSchema(jsonData, dataExtractionSchema, schemaJsonString, options);
                return validatedData;
            } catch (validationError: any) {
                if (validationError instanceof ZodError) {
                    const rawResponseForError = JSON.stringify(jsonData, null, 2);
                    const errorDetails = JSON.stringify(validationError.format(), null, 2);
                    const errorMessage = `Your previous response resulted in an error.
Error Type: SCHEMA_VALIDATION_ERROR
Error Details: ${errorDetails}
The response was valid JSON but did not conform to the required schema. Please review the errors and the schema to provide a corrected response.`;
                    throw new LlmQuerierError(
                        errorMessage,
                        'CUSTOM_ERROR',
                        validationError.format(),
                        rawResponseForError
                    );
                }
                // For other errors, rethrow and let LlmReQuerier handle as critical.
                throw validationError;
            }
        };

        return this.llmReQuerier.query(
            finalMainInstruction,
            userMessagePayload,
            processResponse,
            querierOptions
        );
    }
}
