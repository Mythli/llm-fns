import OpenAI from 'openai'; // Import the OpenAI library
import * as z from "zod";
import { ZodError, ZodTypeAny } from "zod";
import { JsonSchemaClient, JsonSchemaLlmClientOptions } from "./createJsonSchemaLlmClient.js";

export type ZodLlmClientOptions = JsonSchemaLlmClientOptions;

export interface CreateZodLlmClientParams {
    jsonSchemaClient: JsonSchemaClient;
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
    messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[];
    dataExtractionSchema: T;
    options?: ZodLlmClientOptions;
}

export function normalizeZodArgs<T extends ZodTypeAny>(
    arg1: string | OpenAI.Chat.Completions.ChatCompletionMessageParam[] | T,
    arg2?: string | OpenAI.Chat.Completions.ChatCompletionContentPart[] | T | ZodLlmClientOptions,
    arg3?: T | ZodLlmClientOptions,
    arg4?: ZodLlmClientOptions
): NormalizedZodArgs<T> {
    // Case 0: promptZod(messages[], schema, options?)
    if (Array.isArray(arg1)) {
        return {
            messages: arg1 as OpenAI.Chat.Completions.ChatCompletionMessageParam[],
            dataExtractionSchema: arg2 as T,
            options: arg3 as ZodLlmClientOptions | undefined
        };
    }

    if (isZodSchema(arg1)) {
        // Case 1: promptZod(schema, options?)
        return {
            messages: [
                { role: 'system', content: "Generate a valid JSON object based on the schema." },
                { role: 'user', content: "Generate the data." }
            ],
            dataExtractionSchema: arg1,
            options: arg2 as ZodLlmClientOptions | undefined
        };
    }

    if (typeof arg1 === 'string') {
        if (isZodSchema(arg2)) {
            // Case 2: promptZod(prompt, schema, options?)
            return {
                messages: [
                    { role: 'system', content: "You are a helpful assistant that outputs JSON matching the provided schema." },
                    { role: 'user', content: arg1 }
                ],
                dataExtractionSchema: arg2 as T,
                options: arg3 as ZodLlmClientOptions | undefined
            };
        }

        // Case 3: promptZod(system, user, schema, options?)
        return {
            messages: [
                { role: 'system', content: arg1 },
                { role: 'user', content: arg2 as string | OpenAI.Chat.Completions.ChatCompletionContentPart[] }
            ],
            dataExtractionSchema: arg3 as T,
            options: arg4
        };
    }

    throw new Error("Invalid arguments passed to promptZod");
}

export function createZodLlmClient(params: CreateZodLlmClientParams) {
    const { jsonSchemaClient } = params;

    async function promptZod<T extends ZodTypeAny>(
        schema: T,
        options?: ZodLlmClientOptions
    ): Promise<z.infer<T>>;
    async function promptZod<T extends ZodTypeAny>(
        messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
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
        arg1: string | OpenAI.Chat.Completions.ChatCompletionMessageParam[] | T,
        arg2?: string | OpenAI.Chat.Completions.ChatCompletionContentPart[] | T | ZodLlmClientOptions,
        arg3?: T | ZodLlmClientOptions,
        arg4?: ZodLlmClientOptions
    ): Promise<z.infer<T>> {
        const { messages, dataExtractionSchema, options } = normalizeZodArgs(arg1, arg2, arg3, arg4);

        const schema = z.toJSONSchema(dataExtractionSchema, {
            unrepresentable: 'any'
        }) as Record<string, any>;

        const zodValidator = (data: any) => {
            return dataExtractionSchema.parse(data);
        };

        const result = await jsonSchemaClient.promptJson(messages, schema, {
            ...options,
            validator: zodValidator
        });

        return result as z.infer<T>;
    }

    async function isPromptZodCached<T extends ZodTypeAny>(
        schema: T,
        options?: ZodLlmClientOptions
    ): Promise<boolean>;
    async function isPromptZodCached<T extends ZodTypeAny>(
        messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[],
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
        arg1: string | OpenAI.Chat.Completions.ChatCompletionMessageParam[] | T,
        arg2?: string | OpenAI.Chat.Completions.ChatCompletionContentPart[] | T | ZodLlmClientOptions,
        arg3?: T | ZodLlmClientOptions,
        arg4?: ZodLlmClientOptions
    ): Promise<boolean> {
        const { messages, dataExtractionSchema, options } = normalizeZodArgs(arg1, arg2, arg3, arg4);

        const schema = z.toJSONSchema(dataExtractionSchema, {
            unrepresentable: 'any'
        }) as Record<string, any>;

        return jsonSchemaClient.isPromptJsonCached(messages, schema, options);
    }

    return { promptZod, isPromptZodCached };
}

export type ZodLlmClient = ReturnType<typeof createZodLlmClient>;
export type PromptZodFunction = ZodLlmClient['promptZod'];
