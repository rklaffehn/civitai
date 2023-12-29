import { samplerMap } from '~/server/common/constants';
import { ComfyMetaSchema, ImageMetaProps } from '~/server/schema/image.schema';
import { findKeyForValue } from '~/utils/map-helpers';
import { createMetadataProcessor, SDResource } from '~/utils/metadata/base.metadata';
import { fromJson } from '../json-helpers';

const AIR_KEYS = ['ckpt_airs', 'lora_airs', 'embedding_airs'];

function cleanBadJson(str: string) {
  return str.replace(/\[NaN\]/g, '[]').replace(/\[Infinity\]/g, '[]');
}

export const comfyMetadataProcessor = createMetadataProcessor({
  canParse: (exif) => exif.prompt && exif.workflow,
  parse: (exif) => {
    const prompt = JSON.parse(cleanBadJson(exif.prompt as string)) as Record<string, ComfyNode>;
    const samplerNodes: SamplerNode[] = [];
    const models: string[] = [];
    const upscalers: string[] = [];
    const vaes: string[] = [];
    const controlNets: string[] = [];
    const additionalResources: SDResource[] = [];
    const hashes: Record<string, string> = {};

    for (const node of Object.values(prompt)) {
      for (const [key, value] of Object.entries(node.inputs)) {
        if (Array.isArray(value)) node.inputs[key] = prompt[value[0]];
      }

      if (node.class_type == 'KSamplerAdvanced') {
        const simplifiedNode = { ...node.inputs };

        simplifiedNode.steps = getNumberValue(simplifiedNode.steps as ComfyNumber);
        simplifiedNode.cfg = getNumberValue(simplifiedNode.cfg as ComfyNumber);

        samplerNodes.push(simplifiedNode as SamplerNode);
      }

      if (node.class_type == 'KSampler') samplerNodes.push(node.inputs as SamplerNode);

      if (node.class_type == 'LoraLoader') {
        // Ignore lora nodes with strength 0
        const strength = node.inputs.strength_model as number;
        if (strength < 0.001 && strength > -0.001) continue;

        const hash_value = (node.lora_hash as string) || undefined;

        const lora_name = modelFileName(node.inputs.lora_name as string);
        if (hash_value) {
          /* This seems to be what the automatic1111 extension for CivitAI generates for LoRAs. */
          hashes[`lora:${lora_name}`] = hash_value;
        }

        additionalResources.push({
          name: lora_name,
          type: 'lora',
          weight: strength,
          weightClip: node.inputs.strength_clip as number,
          hash: hash_value,
        });
      }

      if (node.class_type == 'CheckpointLoaderSimple') {
        const model_name = modelFileName(node.inputs.ckpt_name as string);
        models.push(model_name);

        const hash_value = (node.ckpt_hash as string) || undefined;

        if (!hashes.model && hash_value) hashes.model = hash_value;

        additionalResources.push({ name: model_name, type: 'model', hash: hash_value });
      }

      if (node.class_type == 'UpscaleModelLoader') upscalers.push(node.inputs.model_name as string);

      if (node.class_type == 'VAELoader') vaes.push(node.inputs.vae_name as string);

      if (node.class_type == 'ControlNetLoader')
        controlNets.push(node.inputs.control_net_name as string);
    }

    const initialSamplerNode =
      samplerNodes.find((x) => x.latent_image.class_type == 'EmptyLatentImage') ?? samplerNodes[0];

    const workflow = JSON.parse(exif.workflow as string) as any;
    const versionIds: number[] = [];
    const modelIds: number[] = [];
    if (workflow?.extra) {
      for (const key of AIR_KEYS) {
        const airs = workflow.extra[key];
        if (!airs) continue;

        for (const air of airs) {
          const [modelId, versionId] = air.split('@');
          if (versionId) versionIds.push(parseInt(versionId));
          else if (modelId) modelIds.push(parseInt(modelId));
        }
      }
    }

    const metadata: ImageMetaProps = {
      prompt: getPromptText(initialSamplerNode.positive),
      negativePrompt: getPromptText(initialSamplerNode.negative),
      cfgScale: initialSamplerNode.cfg,
      steps: initialSamplerNode.steps,
      seed: initialSamplerNode.seed,
      sampler: initialSamplerNode.sampler_name,
      scheduler: initialSamplerNode.scheduler,
      denoise: initialSamplerNode.denoise,
      width: initialSamplerNode.latent_image.inputs.width,
      height: initialSamplerNode.latent_image.inputs.height,
      hashes: hashes,
      models,
      upscalers,
      vaes,
      additionalResources,
      controlNets,
      versionIds,
      modelIds,
      // Converting to string to reduce bytes size
      comfy: JSON.stringify({ prompt, workflow }),
    };

    // Paranoia! If all else fails, the meta data lookup also checks these attributes.
    const model_resource = additionalResources.find((resource) => resource.type === 'model');
    if (model_resource) {
      metadata['Model'] = model_resource.name;
      metadata['Model hash'] = model_resource.hash;
    }

    // Handle control net apply
    if (initialSamplerNode.positive.class_type === 'ControlNetApply') {
      const conditioningNode = initialSamplerNode.positive.inputs.conditioning as ComfyNode;
      metadata.prompt = conditioningNode.inputs.text as string;
    }

    // Map to automatic1111 terms for compatibility
    a1111Compatability(metadata);

    return metadata;
  },
  encode: (meta) => {
    const comfy =
      typeof meta.comfy === 'string' ? fromJson<ComfyMetaSchema>(meta.comfy) : meta.comfy;

    return comfy && comfy.workflow ? JSON.stringify(comfy.workflow) : '';
  },
});

function a1111Compatability(metadata: ImageMetaProps) {
  // Sampler name
  const samplerName = metadata.sampler;
  let a1111sampler: string | undefined;
  if (metadata.scheduler == 'karras') {
    a1111sampler = findKeyForValue(samplerMap, samplerName + '_karras');
  }
  if (!a1111sampler) a1111sampler = findKeyForValue(samplerMap, samplerName);
  if (a1111sampler) metadata.sampler = a1111sampler;

  // Model
  const models = metadata.models as string[];
  if (models.length > 0) {
    metadata.Model = models[0].replace(/\.[^/.]+$/, '');
  }
}

function getPromptText(node: ComfyNode): string {
  if (node.inputs?.text) {
    if (typeof node.inputs.text === 'string') return node.inputs.text;
    if (typeof (node.inputs.text as ComfyNode).class_type !== 'undefined')
      return getPromptText(node.inputs.text as ComfyNode);
  }
  if (node.inputs?.text_g) {
    if (!node.inputs?.text_l || node.inputs?.text_l === node.inputs?.text_g)
      return node.inputs.text_g as string;
    return `${node.inputs.text_g}, ${node.inputs.text_l}`;
  }
  return '';
}

type ComfyNumber = ComfyNode | number;
function getNumberValue(input: ComfyNumber) {
  if (typeof input === 'number') return input;
  return input.inputs.Value as number;
}

/** ComfyUI can handle models in sub-directories but we need the raw filename here.
 *
 * @param name The model's name, possibly with a relative path prefix
 * @returns The model's filename only
 */
function modelFileName(name: string): string {
  const sep_expr = /\\(\\\\)*/g;
  name = name.replace(sep_expr, '/');
  const parts = name.split('/');
  return parts[parts.length - 1].replace(/\.[^/.]+$/, '');
}

// #region [types]
type ComfyNode = {
  inputs: Record<string, number | string | Array<string | number> | ComfyNode>;
  class_type: string;
  /* ComfyUI must somehow transport model hashes (first ten digits of the file's sha256) in its prompt. One solution
     could be adding them to the load nodes from where we also pick the model names. */
  ckpt_hash?: string;
  lora_hash?: string;
};

type SamplerNode = {
  seed: number;
  steps: number;
  cfg: number;
  sampler_name: string;
  scheduler: string;
  denoise: number;
  model: ComfyNode;
  positive: ComfyNode;
  negative: ComfyNode;
  latent_image: ComfyNode;
};
// #endregion
