import { GoogleGenAI, Schema, Type } from "@google/genai";
import { NodeType, TaskType } from "../types";

const getGeminiClient = () => {
  const apiKey = import.meta.env.VITE_API_KEY;
  if (!apiKey) {
    throw new Error("Missing Gemini API key. Add VITE_API_KEY to your .env file.");
  }
  return new GoogleGenAI({ apiKey });
};

// Helper to calculate positions for a generated graph so nodes aren't stacked
const layoutNodes = (nodes: any[]) => {
  let x = 100;
  const y = 250;
  return nodes.map((node, index) => {
    // If the AI didn't give a position, assign one linearly
    const newNode = {
      ...node,
      position: node.position || { x: x + (index * 350), y }
    };
    return newNode;
  });
};

export const generatePipeline = async (prompt: string): Promise<{ nodes: any[], edges: any[], explanation: string }> => {
  const ai = getGeminiClient();

  const systemInstruction = `
    You are an AI Architect for NeuroFlow, a node-based AI workflow editor.
    Your goal is to convert user requests into a valid JSON graph structure with 'nodes' and 'edges'.

    Supported Node Types:
    - 'dataset': Input data. Config: { path: string, split: string, source_type: 'hub'|'local' }
    - 'model': AI Model. Config: { modelId: string, quantization: 'none'|'int4'|'int8' }
    - 'task': The objective. Config: { task: string (e.g. 'llm-sft', 'text-classification', 'image-generation') }
    - 'trainer': Training settings. Config: { learning_rate: number, epochs: number, use_peft: boolean }
    - 'inference': Run a model. Config: { provider: 'openai'|'gemini', model: string }
    - 'prompt': Prompt template. Config: { template: string }
    - 'router': Conditional logic. Config: { condition_type: 'contains', match_value: string }
    - 'google_search': Search tool. Config: { query: string }
    - 'python_repl': Code execution. Config: { code: string }
    - 'deploy': Push to hub. Config: { hubId: string }
    
    Rules:
    1. Always connect nodes logically (e.g., Dataset -> Model -> Task -> Trainer).
    2. Use the exact 'type' strings listed above.
    3. Generate unique IDs for nodes (e.g., 'node-1', 'node-2').
    4. Provide a brief 'explanation' of what you built.
  `;

  // Define the schema to ensure strict JSON output
  const responseSchema: Schema = {
    type: Type.OBJECT,
    properties: {
      nodes: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            id: { type: Type.STRING },
            type: { type: Type.STRING },
            data: {
              type: Type.OBJECT,
              properties: {
                label: { type: Type.STRING },
                type: { type: Type.STRING },
                // Removing the strict OBJECT type for config as it causes errors when empty
                // The AI will still generate it based on the system instructions
              },
              required: ['label', 'type']
            },
            position: {
              type: Type.OBJECT,
              properties: {
                x: { type: Type.NUMBER },
                y: { type: Type.NUMBER }
              },
              nullable: true
            }
          },
          required: ['id', 'type', 'data']
        }
      },
      edges: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            id: { type: Type.STRING },
            source: { type: Type.STRING },
            target: { type: Type.STRING }
          },
          required: ['id', 'source', 'target']
        }
      },
      explanation: { type: Type.STRING }
    },
    required: ['nodes', 'edges', 'explanation']
  };

  try {
    // @google/genai requires contents to be an array of Content objects
    // and systemInstruction is passed in the config
    const response = await ai.models.generateContent({
      model: 'gemini-pro',
      contents: [
        {
          role: 'user',
          parts: [{ text: prompt }]
        }
      ],
      config: {
        systemInstruction: systemInstruction,
        responseMimeType: "application/json",
        responseSchema: responseSchema,
        temperature: 0.1,
      },
    });

    // Extract text from candidates in @google/genai
    const text = response.candidates?.[0]?.content?.parts?.[0]?.text;

    if (!text) {
      console.error("Gemini Response Details:", JSON.stringify(response, null, 2));
      throw new Error("AI returned an empty or malformed response.");
    }

    const result = JSON.parse(text);

    // Post-process layout
    result.nodes = layoutNodes(result.nodes);

    return result;

  } catch (error: any) {
    console.error("AI Generation Detailed Error:", error);
    const msg = error.message || (typeof error === 'string' ? error : "Unknown error");
    throw new Error(`AI Build Error: ${msg}`);
  }
};

export const chatWithAI = async (prompt: string): Promise<string> => {
  const ai = getGeminiClient();
  try {
    const response = await ai.models.generateContent({
      model: 'gemini-pro',
      contents: [{ role: 'user', parts: [{ text: prompt }] }],
      config: {
        systemInstruction: "You are NeuroBot, a helpful AI Architect assistant for the NeuroFlow platform. Be concise and friendly. If the user asks to build or generate a pipeline, tell them what you can do.",
        temperature: 0.7,
      }
    });

    return response.candidates?.[0]?.content?.parts?.[0]?.text || "I'm sorry, I couldn't process that.";
  } catch (error: any) {
    console.error("Chat Error:", error);
    return `Error: ${error.message || 'Unknown error'}`;
  }
};
