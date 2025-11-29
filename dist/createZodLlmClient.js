"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.normalizeZodArgs = normalizeZodArgs;
exports.createZodLlmClient = createZodLlmClient;
const z = __importStar(require("zod"));
const createLlmRetryClient_js_1 = require("./createLlmRetryClient.js");
const zod_1 = require("zod");
function isZodSchema(obj) {
    return (typeof obj === 'object' &&
        obj !== null &&
        'parse' in obj &&
        '_def' in obj);
}
function normalizeZodArgs(arg1, arg2, arg3, arg4) {
    if (isZodSchema(arg1)) {
        // Case 1: promptZod(schema, options?)
        return {
            mainInstruction: "Generate a valid JSON object based on the schema.",
            userMessagePayload: "Generate the data.",
            dataExtractionSchema: arg1,
            options: arg2
        };
    }
    if (typeof arg1 === 'string') {
        if (isZodSchema(arg2)) {
            // Case 2: promptZod(prompt, schema, options?)
            return {
                mainInstruction: "You are a helpful assistant that outputs JSON matching the provided schema.",
                userMessagePayload: arg1,
                dataExtractionSchema: arg2,
                options: arg3
            };
        }
        // Case 3: promptZod(system, user, schema, options?)
        return {
            mainInstruction: arg1,
            userMessagePayload: arg2,
            dataExtractionSchema: arg3,
            options: arg4
        };
    }
    throw new Error("Invalid arguments passed to promptZod");
}
function createZodLlmClient(params) {
    const { prompt, isPromptCached, fallbackPrompt, disableJsonFixer = false } = params;
    const llmRetryClient = (0, createLlmRetryClient_js_1.createLlmRetryClient)({ prompt, fallbackPrompt });
    async function _tryToFixJson(brokenResponse, schemaJsonString, errorDetails, options) {
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
        const messages = [
            { role: 'system', content: 'You are an expert at fixing malformed JSON data to match a specific schema.' },
            { role: 'user', content: fixupPrompt }
        ];
        const useResponseFormat = options?.useResponseFormat ?? true;
        const response_format = useResponseFormat
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
    async function _parseOrFixJson(llmResponseString, schemaJsonString, options) {
        let jsonDataToParse = llmResponseString.trim();
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
        }
        catch (parseError) {
            if (disableJsonFixer) {
                throw parseError; // re-throw original error
            }
            // Attempt a one-time fix before failing.
            const errorDetails = `JSON Parse Error: ${parseError.message}`;
            const fixedResponse = await _tryToFixJson(jsonDataToParse, schemaJsonString, errorDetails, options);
            if (fixedResponse) {
                try {
                    return JSON.parse(fixedResponse);
                }
                catch (e) {
                    // Fix-up failed, throw original error.
                    throw parseError;
                }
            }
            throw parseError; // if no fixed response
        }
    }
    async function _validateOrFixSchema(jsonData, dataExtractionSchema, schemaJsonString, options) {
        try {
            if (options?.beforeValidation) {
                jsonData = options.beforeValidation(jsonData);
            }
            return dataExtractionSchema.parse(jsonData);
        }
        catch (validationError) {
            if (!(validationError instanceof zod_1.ZodError) || disableJsonFixer) {
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
                }
                catch (e) {
                    // Fix-up failed, throw original validation error
                    throw validationError;
                }
            }
            throw validationError; // if no fixed response
        }
    }
    function _getZodPromptConfig(mainInstruction, dataExtractionSchema, options) {
        const schema = z.toJSONSchema(dataExtractionSchema, {
            unrepresentable: 'any'
        });
        const schemaJsonString = JSON.stringify(schema);
        const commonPromptFooter = `
Your response MUST be a single JSON entity (object or array) that strictly adheres to the following JSON schema.
Do NOT include any other text, explanations, or markdown formatting (like \`\`\`json) before or after the JSON entity.

JSON schema:
${schemaJsonString}`;
        const finalMainInstruction = `${mainInstruction}\n${commonPromptFooter}`;
        const useResponseFormat = options?.useResponseFormat ?? true;
        const response_format = useResponseFormat
            ? { type: 'json_object' }
            : undefined;
        return { finalMainInstruction, schemaJsonString, response_format };
    }
    async function promptZod(arg1, arg2, arg3, arg4) {
        const { mainInstruction, userMessagePayload, dataExtractionSchema, options } = normalizeZodArgs(arg1, arg2, arg3, arg4);
        const { finalMainInstruction, schemaJsonString, response_format } = _getZodPromptConfig(mainInstruction, dataExtractionSchema, options);
        const processResponse = async (llmResponseString) => {
            let jsonData;
            try {
                jsonData = await _parseOrFixJson(llmResponseString, schemaJsonString, options);
            }
            catch (parseError) {
                const errorMessage = `Your previous response resulted in an error.
Error Type: JSON_PARSE_ERROR
Error Details: ${parseError.message}
The response provided was not valid JSON. Please correct it.`;
                throw new createLlmRetryClient_js_1.LlmRetryError(errorMessage, 'JSON_PARSE_ERROR', undefined, llmResponseString);
            }
            try {
                const validatedData = await _validateOrFixSchema(jsonData, dataExtractionSchema, schemaJsonString, options);
                return validatedData;
            }
            catch (validationError) {
                if (validationError instanceof zod_1.ZodError) {
                    const rawResponseForError = JSON.stringify(jsonData, null, 2);
                    const errorDetails = JSON.stringify(validationError.format(), null, 2);
                    const errorMessage = `Your previous response resulted in an error.
Error Type: SCHEMA_VALIDATION_ERROR
Error Details: ${errorDetails}
The response was valid JSON but did not conform to the required schema. Please review the errors and the schema to provide a corrected response.`;
                    throw new createLlmRetryClient_js_1.LlmRetryError(errorMessage, 'CUSTOM_ERROR', validationError.format(), rawResponseForError);
                }
                // For other errors, rethrow and let LlmRetryClient handle as critical.
                throw validationError;
            }
        };
        const messages = [
            { role: "system", content: finalMainInstruction },
            { role: "user", content: userMessagePayload }
        ];
        const retryOptions = {
            ...options,
            messages,
            response_format,
            validate: processResponse
        };
        // Use promptTextRetry because we expect a string response to parse as JSON
        return llmRetryClient.promptTextRetry(retryOptions);
    }
    async function isPromptZodCached(arg1, arg2, arg3, arg4) {
        const { mainInstruction, userMessagePayload, dataExtractionSchema, options } = normalizeZodArgs(arg1, arg2, arg3, arg4);
        const { finalMainInstruction, response_format } = _getZodPromptConfig(mainInstruction, dataExtractionSchema, options);
        const messages = [
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
