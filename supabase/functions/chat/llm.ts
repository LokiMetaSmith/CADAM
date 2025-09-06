import { Anthropic } from 'https://esm.sh/@anthropic-ai/sdk@0.53.0';
import {
  MessageCreateParams,
  MessageParam,
} from 'https://esm.sh/@anthropic-ai/sdk@0.53.0/resources/messages.d.mts';
import { Model } from '@shared/types.ts';
import OpenAI from 'https://esm.sh/openai@4.33.0';
import {
  GoogleGenerativeAI,
  HarmCategory,
  HarmBlockThreshold,
} from 'npm:@google/generative-ai';
import Groq from 'https://esm.sh/groq-sdk@0.3.2';

export interface LlmClient {
  create(
    params: MessageCreateParams,
    model: string,
  ): Promise<ReadableStream<any>>;
  createNonStreaming(
    params: MessageCreateParams,
    model: string,
  ): Promise<any>;
  generateTitle(messages: MessageParam[]): Promise<string>;
  parseStream(
    stream: ReadableStream<any>,
  ): ReadableStream<any>;
}

class AnthropicClient implements LlmClient {
  private anthropic: Anthropic;

  constructor(apiKey: string) {
    this.anthropic = new Anthropic({ apiKey });
  }

  create(
    params: MessageCreateParams,
    model: string,
  ): Promise<ReadableStream<any>> {
    return this.anthropic.messages.create({ ...params, model, stream: true });
  }

  createNonStreaming(
    params: MessageCreateParams,
    model: string,
  ): Promise<any> {
    return this.anthropic.messages.create({ ...params, model, stream: false });
  }

  async generateTitle(messages: MessageParam[]): Promise<string> {
    try {
      const titleSystemPrompt = `You are a helpful assistant that generates concise, descriptive titles for 3D objects based on a user's description, conversation context, and any reference images. Your titles should be:
1. Brief (under 27 characters)
2. Descriptive of the object
3. Clear and professional
4. Without any special formatting or punctuation at the beginning or end
5. Consider the entire conversation context, not just the latest message
6. When images are provided, incorporate visual elements you can see into the title`;

      const titleResponse = await this.anthropic.messages.create({
        model: 'claude-3-haiku-20240307',
        max_tokens: 100,
        system: titleSystemPrompt,
        messages: [
          ...messages,
          {
            role: 'user',
            content:
              'Generate a concise title for the 3D object that will be generated based on the previous messages.',
          },
        ],
      });

      if (
        Array.isArray(titleResponse.content) &&
        titleResponse.content.length > 0
      ) {
        const lastContent =
          titleResponse.content[titleResponse.content.length - 1];
        if (lastContent.type === 'text') {
          let title = lastContent.text.trim();
          if (title.length > 60) title = title.substring(0, 57) + '...';
          return title;
        }
      }
    } catch (error) {
      console.error('Error generating object title:', error);
    }
    return 'Adam Object';
  }

  parseStream(stream: ReadableStream<any>): ReadableStream<any> {
    const reader = stream.getReader();
    return new ReadableStream({
      async start(controller) {
        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            controller.close();
            break;
          }
          const chunk = decoder.decode(value);
          const lines = chunk
            .split('\n')
            .filter((line) => line.trim() !== '');
          for (const line of lines) {
            const message = line.replace(/^data: /, '');
            if (message === '[DONE]') {
              controller.close();
              return;
            }
            try {
              const parsed = JSON.parse(message);
              controller.enqueue(parsed);
            } catch (error) {
              console.error('Error parsing stream chunk:', error);
            }
          }
        }
      },
    });
  }
}

class GrokClient implements LlmClient {
  private groq: Groq;

  constructor(apiKey: string) {
    this.groq = new Groq({ apiKey });
  }

  create(
    params: MessageCreateParams,
    model: string,
  ): Promise<ReadableStream<any>> {
    const messages = params.messages.map((m) => {
      if (typeof m.content !== 'string') {
        throw new Error('GrokClient does not support multimodal content.');
      }
      return {
        role: m.role,
        content: m.content,
      };
    });

    const groqParams: Groq.Chat.CompletionCreateParams = {
      messages,
      model,
    };

    if (params.tools) {
      groqParams.tools = params.tools;
      groqParams.tool_choice = 'auto';
    }

    return this.groq.chat.completions.create(groqParams);
  }

  createNonStreaming(
    params: MessageCreateParams,
    model: string,
  ): Promise<any> {
    const messages = params.messages.map((m) => {
      if (typeof m.content !== 'string') {
        throw new Error('GrokClient does not support multimodal content.');
      }
      return {
        role: m.role,
        content: m.content,
      };
    });

    const groqParams: Groq.Chat.CompletionCreateParams = {
      messages,
      model,
    };

    if (params.tools) {
      groqParams.tools = params.tools;
      groqParams.tool_choice = 'auto';
    }

    return this.groq.chat.completions.create({ ...groqParams, stream: false });
  }

  async generateTitle(messages: MessageParam[]): Promise<string> {
    try {
      const titleSystemPrompt = `You are a helpful assistant that generates concise, descriptive titles for 3D objects based on a user's description, conversation context, and any reference images. Your titles should be:
1. Brief (under 27 characters)
2. Descriptive of the object
3. Clear and professional
4. Without any special formatting or punctuation at the beginning or end
5. Consider the entire conversation context, not just the latest message
6. When images are provided, incorporate visual elements you can see into the title`;

      const titleResponse = await this.groq.chat.completions.create({
        model: 'llama3-8b-8192',
        messages: [
          { role: 'system', content: titleSystemPrompt },
          ...messages,
          {
            role: 'user',
            content:
              'Generate a concise title for the 3D object that will be generated based on the previous messages.',
          },
        ],
      });

      const title = titleResponse.choices[0]?.message?.content?.trim();
      if (title) {
        return title.length > 60 ? title.substring(0, 57) + '...' : title;
      }
    } catch (error) {
      console.error('Error generating object title:', error);
    }
    return 'Adam Object';
  }

  parseStream(stream: ReadableStream<any>): ReadableStream<any> {
    const reader = stream.getReader();
    return new ReadableStream({
      async start(controller) {
        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            controller.close();
            break;
          }
          const chunk = decoder.decode(value);
          const lines = chunk
            .split('\n')
            .filter((line) => line.trim() !== '');
          for (const line of lines) {
            const message = line.replace(/^data: /, '');
            if (message === '[DONE]') {
              controller.close();
              return;
            }
            try {
              const parsed = JSON.parse(message);
              controller.enqueue(parsed);
            } catch (error) {
              console.error('Error parsing stream chunk:', error);
            }
          }
        }
      },
    });
  }
}

class GoogleClient implements LlmClient {
  private google: GoogleGenerativeAI;

  constructor(apiKey: string) {
    this.google = new GoogleGenerativeAI(apiKey);
  }

  async create(
    params: MessageCreateParams,
    model: string,
  ): Promise<ReadableStream<any>> {
    const googleModel = this.google.getGenerativeModel({ model });

    const history = await Promise.all(
      params.messages.map(async (m) => {
        if (typeof m.content === 'string') {
          return { role: m.role, parts: [{ text: m.content }] };
        }

        const parts = await Promise.all(
          m.content.map(async (c) => {
            if (c.type === 'text') {
              return { text: c.text };
            } else if (c.type === 'image' && c.source.type === 'url') {
              const response = await fetch(c.source.url);
              const image = await response.arrayBuffer();
              return {
                inlineData: {
                  data: btoa(String.fromCharCode(...new Uint8Array(image))),
                  mimeType: response.headers.get('content-type')!,
                },
              };
            }
            return { text: '' };
          }),
        );

        return { role: m.role, parts };
      }),
    );

    const generationConfig: any = {
      maxOutputTokens: params.max_tokens,
    };

    if (params.tools) {
      generationConfig.tools = [{ functionDeclarations: params.tools }];
    }

    const chat = googleModel.startChat({
      history,
      generationConfig,
      safetySettings: [
        {
          category: HarmCategory.HARM_CATEGORY_HARASSMENT,
          threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
      ],
    });

    const lastMessage = params.messages[params.messages.length - 1];
    const lastMessageContent =
      typeof lastMessage.content === 'string'
        ? lastMessage.content
        : lastMessage.content
            .filter((c) => c.type === 'text')
            .map((c: any) => c.text)
            .join(' ');

    const result = await chat.sendMessageStream(lastMessageContent);
    return result.stream;
  }

  async createNonStreaming(
    params: MessageCreateParams,
    model: string,
  ): Promise<any> {
    const googleModel = this.google.getGenerativeModel({ model });

    const history = await Promise.all(
      params.messages.map(async (m) => {
        if (typeof m.content === 'string') {
          return { role: m.role, parts: [{ text: m.content }] };
        }

        const parts = await Promise.all(
          m.content.map(async (c) => {
            if (c.type === 'text') {
              return { text: c.text };
            } else if (c.type === 'image' && c.source.type === 'url') {
              const response = await fetch(c.source.url);
              const image = await response.arrayBuffer();
              return {
                inlineData: {
                  data: btoa(String.fromCharCode(...new Uint8Array(image))),
                  mimeType: response.headers.get('content-type')!,
                },
              };
            }
            return { text: '' };
          }),
        );

        return { role: m.role, parts };
      }),
    );

    const generationConfig: any = {
      maxOutputTokens: params.max_tokens,
    };

    if (params.tools) {
      generationConfig.tools = [{ functionDeclarations: params.tools }];
    }

    const chat = googleModel.startChat({
      history,
      generationConfig,
      safetySettings: [
        {
          category: HarmCategory.HARM_CATEGORY_HARASSMENT,
          threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
      ],
    });

    const lastMessage = params.messages[params.messages.length - 1];
    const lastMessageContent =
      typeof lastMessage.content === 'string'
        ? lastMessage.content
        : lastMessage.content
            .filter((c) => c.type === 'text')
            .map((c: any) => c.text)
            .join(' ');

    return chat.sendMessage(lastMessageContent);
  }

  async generateTitle(messages: MessageParam[]): Promise<string> {
    try {
      const titleSystemPrompt = `You are a helpful assistant that generates concise, descriptive titles for 3D objects based on a user's description, conversation context, and any reference images. Your titles should be:
1. Brief (under 27 characters)
2. Descriptive of the object
3. Clear and professional
4. Without any special formatting or punctuation at the beginning or end
5. Consider the entire conversation context, not just the latest message
6. When images are provided, incorporate visual elements you can see into the title`;

      const googleModel = this.google.getGenerativeModel({
        model: 'gemini-pro',
      });
      const chat = googleModel.startChat({
        history: [
          { role: 'user', parts: [{ text: titleSystemPrompt }] },
          { role: 'model', parts: [{ text: 'OK' }] },
          ...messages.map((m) => ({
            role: m.role,
            parts: [{ text: m.content }],
          })),
        ],
      });
      const result = await chat.sendMessage(
        'Generate a concise title for the 3D object that will be generated based on the previous messages.',
      );
      const title = result.response.text().trim();
      if (title) {
        return title.length > 60 ? title.substring(0, 57) + '...' : title;
      }
    } catch (error) {
      console.error('Error generating object title:', error);
    }
    return 'Adam Object';
  }

  parseStream(stream: ReadableStream<any>): ReadableStream<any> {
    const reader = stream.getReader();
    return new ReadableStream({
      async start(controller) {
        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            controller.close();
            break;
          }
          const chunk = decoder.decode(value);
          const lines = chunk
            .split('\n')
            .filter((line) => line.trim() !== '');
          for (const line of lines) {
            const message = line.replace(/^data: /, '');
            if (message === '[DONE]') {
              controller.close();
              return;
            }
            try {
              const parsed = JSON.parse(message);
              controller.enqueue(parsed);
            } catch (error) {
              console.error('Error parsing stream chunk:', error);
            }
          }
        }
      },
    });
  }
}

class LlamaClient implements LlmClient {
  private openai: OpenAI;

  constructor(baseURL: string) {
    this.openai = new OpenAI({ baseURL });
  }

  create(
    params: MessageCreateParams,
    model: string,
  ): Promise<ReadableStream<any>> {
    const messages = params.messages.map((m) => {
      if (typeof m.content !== 'string') {
        throw new Error('LlamaClient does not support multimodal content.');
      }
      return {
        role: m.role,
        content: m.content,
      };
    });

    const llamaParams: OpenAI.Chat.CompletionCreateParams = {
      messages,
      model,
    };

    if (params.tools) {
      llamaParams.tools = params.tools;
      llamaParams.tool_choice = 'auto';
    }

    return this.openai.chat.completions.create(llamaParams);
  }

  createNonStreaming(
    params: MessageCreateParams,
    model: string,
  ): Promise<any> {
    const messages = params.messages.map((m) => {
      if (typeof m.content !== 'string') {
        throw new Error('LlamaClient does not support multimodal content.');
      }
      return {
        role: m.role,
        content: m.content,
      };
    });

    const llamaParams: OpenAI.Chat.CompletionCreateParams = {
      messages,
      model,
    };

    if (params.tools) {
      llamaParams.tools = params.tools;
      llamaParams.tool_choice = 'auto';
    }

    return this.openai.chat.completions.create({ ...llamaParams, stream: false });
  }

  async generateTitle(messages: MessageParam[]): Promise<string> {
    try {
      const titleSystemPrompt = `You are a helpful assistant that generates concise, descriptive titles for 3D objects based on a user's description, conversation context, and any reference images. Your titles should be:
1. Brief (under 27 characters)
2. Descriptive of the object
3. Clear and professional
4. Without any special formatting or punctuation at the beginning or end
5. Consider the entire conversation context, not just the latest message
6. When images are provided, incorporate visual elements you can see into the title`;

      const titleResponse = await this.openai.chat.completions.create({
        model: 'llama3',
        messages: [
          { role: 'system', content: titleSystemPrompt },
          ...messages,
          {
            role: 'user',
            content:
              'Generate a concise title for the 3D object that will be generated based on the previous messages.',
          },
        ],
      });

      const title = titleResponse.choices[0]?.message?.content?.trim();
      if (title) {
        return title.length > 60 ? title.substring(0, 57) + '...' : title;
      }
    } catch (error) {
      console.error('Error generating object title:', error);
    }
    return 'Adam Object';
  }

  parseStream(stream: ReadableStream<any>): ReadableStream<any> {
    const reader = stream.getReader();
    return new ReadableStream({
      async start(controller) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            controller.close();
            break;
          }
          const chunk = {
            type: 'content_block_delta',
            delta: {
              type: 'text_delta',
              text: value.response.text(),
            },
          };
          controller.enqueue(chunk);
        }
      },
    });
  }
}

export function getLlmClient(model: Model): LlmClient {
  switch (model) {
    case 'anthropic-fast':
    case 'anthropic-quality':
      return new AnthropicClient(Deno.env.get('ANTHROPIC_API_KEY') ?? '');
    case 'grok':
      return new GrokClient(Deno.env.get('GROK_API_KEY') ?? '');
    case 'google':
      return new GoogleClient(Deno.env.get('GOOGLE_API_KEY') ?? '');
    case 'llama':
      return new LlamaClient(Deno.env.get('LLAMA_API_URL') ?? '');
    default:
      throw new Error(`Unsupported model: ${model}`);
  }
}
