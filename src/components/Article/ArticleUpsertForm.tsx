import {
  Anchor,
  Button,
  ButtonProps,
  Group,
  Stack,
  StackProps,
  Text,
  ThemeIcon,
  Title,
  Tooltip,
  TooltipProps,
  createStyles,
  ActionIcon,
  Paper,
  Divider,
  Checkbox,
} from '@mantine/core';
import { TagTarget } from '@prisma/client';
import { IconQuestionMark, IconTrash } from '@tabler/icons-react';
import { useRouter } from 'next/router';
import React, { useState } from 'react';
import { z } from 'zod';

import { BackButton } from '~/components/BackButton/BackButton';
import { hiddenLabel, matureLabel } from '~/components/Post/Edit/EditPostControls';
import { useFormStorage } from '~/hooks/useFormStorage';
import {
  Form,
  InputCheckbox,
  InputClubResourceManagementInput,
  InputMultiFileUpload,
  InputRTE,
  InputSelect,
  InputSimpleImageUpload,
  InputTags,
  InputText,
  useForm,
} from '~/libs/form';
import { hideMobile, showMobile } from '~/libs/sx-helpers';
import { upsertArticleInput } from '~/server/schema/article.schema';
import { ArticleGetById } from '~/types/router';
import { formatDate } from '~/utils/date-helpers';
import { showErrorNotification } from '~/utils/notifications';
import { parseNumericString } from '~/utils/query-string-helpers';
import { titleCase } from '~/utils/string-helpers';
import { trpc } from '~/utils/trpc';
import { ContainerGrid } from '~/components/ContainerGrid/ContainerGrid';
import { useFeatureFlags } from '~/providers/FeatureFlagsProvider';
import { useQueryUserContributingClubs } from '../Club/club.utils';

const schema = upsertArticleInput.extend({
  categoryId: z.number().min(0, 'Please select a valid category'),
  cover: z
    .preprocess(
      (val) => (typeof val === 'string' ? { url: val } : val),
      z.object({ url: z.string().nonempty() })
    )
    .refine((data) => !!data.url, { message: 'Please upload a cover image' }),
});
const querySchema = z.object({
  category: z.preprocess(parseNumericString, z.number().optional().default(-1)),
});

const tooltipProps: Partial<TooltipProps> = {
  maw: 300,
  multiline: true,
  position: 'bottom',
  withArrow: true,
};

const useStyles = createStyles((theme) => ({
  sidebar: {
    position: 'sticky',
    top: 70 + theme.spacing.xl,
  },
}));

export function ArticleUpsertForm({ article }: Props) {
  const { classes } = useStyles();
  const queryUtils = trpc.useContext();
  const router = useRouter();
  const result = querySchema.safeParse(router.query);
  const features = useFeatureFlags();
  const { hasClubs } = useQueryUserContributingClubs();

  const defaultCategory = result.success ? result.data.category : -1;
  const defaultValues: z.infer<typeof schema> = {
    ...article,
    title: article?.title ?? '',
    content: article?.content ?? '',
    categoryId: article?.tags.find((tag) => tag.isCategory)?.id ?? defaultCategory,
    tags: article?.tags.filter((tag) => !tag.isCategory) ?? [],
    cover: article?.cover ? { url: article.cover ?? '' } : { url: '' },
    clubs: article?.clubs ?? [],
    unlisted: article?.unlisted ?? false,
  };
  const form = useForm({ schema, defaultValues, shouldUnregister: false });
  const clearStorage = useFormStorage({
    schema,
    form,
    timeout: 1000,
    key: `article${article?.id ? `_${article?.id}` : 'new'}`,
    watch: ({ content, cover, categoryId, nsfw, tags, title, unlisted }) => ({
      content,
      cover,
      categoryId,
      nsfw,
      tags,
      title,
      unlisted,
    }),
  });

  const [unlisted] = form.watch(['unlisted']);
  const [publishing, setPublishing] = useState(false);

  const { data, isLoading: loadingCategories } = trpc.tag.getAll.useQuery({
    categories: true,
    entityType: [TagTarget.Article],
    unlisted: false,
    limit: 100,
  });
  const categories =
    data?.items.map((tag) => ({ label: titleCase(tag.name), value: tag.id })) ?? [];

  const upsertArticleMutation = trpc.article.upsert.useMutation();

  const handleSubmit = ({
    categoryId,
    tags: selectedTags,
    cover,
    ...rest
  }: z.infer<typeof schema>) => {
    const selectedCategory = data?.items.find((cat) => cat.id === categoryId);
    const tags =
      selectedTags && selectedCategory ? selectedTags.concat([selectedCategory]) : selectedTags;
    upsertArticleMutation.mutate(
      { ...rest, tags, publishedAt: publishing ? new Date() : null, cover: cover.url },
      {
        async onSuccess(result) {
          await router.push(`/articles/${result.id}`);
          await queryUtils.article.getById.invalidate({ id: result.id });
          await queryUtils.article.getInfinite.invalidate();
          await queryUtils.article.getByCategory.invalidate();
          clearStorage();
        },
        onError(error) {
          showErrorNotification({
            title: 'Failed to save article',
            error: new Error(error.message),
          });
        },
      }
    );
  };

  return (
    <Form form={form} onSubmit={handleSubmit}>
      <ContainerGrid gutter="xl">
        <ContainerGrid.Col xs={12} md={8}>
          <Stack spacing="xl">
            <Group spacing={4}>
              <BackButton url="/articles" />
              <Title>{article?.id ? 'Editing article' : 'Create an Article'}</Title>
            </Group>
            <InputText
              name="title"
              label="Title"
              placeholder="e.g.: How to create your own LoRA"
              withAsterisk
            />
            <InputRTE
              name="content"
              label="Content"
              editorSize="xl"
              includeControls={[
                'heading',
                'formatting',
                'list',
                'link',
                'media',
                'polls',
                'colors',
              ]}
              withAsterisk
              stickyToolbar
            />
            {features.clubs && hasClubs && (
              <>
                <Divider label="Club options" />
                <InputClubResourceManagementInput
                  label="Make this resource part of a club"
                  description="This will make this resource available to club members only. People will still see this resource in the public list, but will be required to join the club to use it."
                  name="clubs"
                />
              </>
            )}
          </Stack>
        </ContainerGrid.Col>
        <ContainerGrid.Col xs={12} md={4}>
          <Stack className={classes.sidebar} spacing="xl">
            <ActionButtons
              article={article}
              saveButtonProps={{
                loading: upsertArticleMutation.isLoading && !publishing,
                disabled: upsertArticleMutation.isLoading,
                onClick: () => setPublishing(false),
              }}
              publishButtonProps={{
                loading: upsertArticleMutation.isLoading && publishing,
                disabled: upsertArticleMutation.isLoading,
                onClick: () => setPublishing(true),
              }}
              sx={hideMobile}
            />
            <InputCheckbox
              name="nsfw"
              label={
                <Group spacing={4}>
                  Mature
                  <Tooltip label={matureLabel} {...tooltipProps}>
                    <ThemeIcon radius="xl" size="xs" color="gray">
                      <IconQuestionMark />
                    </ThemeIcon>
                  </Tooltip>
                </Group>
              }
            />
            {features.clubs && hasClubs && (
              <>
                <Checkbox
                  checked={!unlisted}
                  onChange={() => {
                    form.setValue('unlisted', !unlisted);
                  }}
                  label={
                    <Stack>
                      <Text>Visible to everyone</Text>
                    </Stack>
                  }
                />
                <Text size="xs">
                  By marking this visible to everyone, a preview of this article will be displayed
                  in the article feed for other users. If you want this article to only be visible
                  to club members, uncheck this.
                </Text>
              </>
            )}
            <InputSimpleImageUpload name="cover" label="Cover Image" withAsterisk />
            <InputSelect
              name="categoryId"
              label="Category"
              placeholder="Select a category"
              data={categories}
              nothingFound="Nothing found"
              loading={loadingCategories}
              withAsterisk
            />
            <InputTags
              name="tags"
              label="Tags"
              target={[TagTarget.Article]}
              filter={(tag) =>
                data && tag.name ? !data.items.map((cat) => cat.name).includes(tag.name) : true
              }
            />
            <InputMultiFileUpload
              name="attachments"
              label="Attachments"
              dropzoneProps={{
                maxSize: 30 * 1024 ** 2, // 30MB
                maxFiles: 10,
                accept: {
                  'application/pdf': ['.pdf'],
                  'application/zip': ['.zip'],
                  'application/json': ['.json'],
                  'application/x-yaml': ['.yaml', '.yml'],
                  'text/plain': ['.txt'],
                  'text/markdown': ['.md'],
                  'text/x-python-script': ['.py'],
                },
              }}
              renderItem={(file, onRemove) => (
                <Paper key={file.id} radius="sm" p={0} w="100%">
                  <Group position="apart">
                    {article && file.id ? (
                      <Anchor href={`/api/download/attachments/${file.id}`} lineClamp={1} download>
                        {file.name}
                      </Anchor>
                    ) : (
                      <Text size="sm" weight={500} lineClamp={1}>
                        {file.name}
                      </Text>
                    )}
                    <Tooltip label="Remove">
                      <ActionIcon size="sm" color="red" variant="transparent" onClick={onRemove}>
                        <IconTrash />
                      </ActionIcon>
                    </Tooltip>
                  </Group>
                </Paper>
              )}
            />
            <ActionButtons
              article={article}
              saveButtonProps={{
                loading: upsertArticleMutation.isLoading && !publishing,
                disabled: upsertArticleMutation.isLoading,
                onClick: () => setPublishing(false),
              }}
              publishButtonProps={{
                loading: upsertArticleMutation.isLoading && publishing,
                disabled: upsertArticleMutation.isLoading,
                onClick: () => setPublishing(true),
              }}
              sx={showMobile}
            />
          </Stack>
        </ContainerGrid.Col>
      </ContainerGrid>
    </Form>
  );
}

type Props = { article?: ArticleGetById };

function ActionButtons({
  article,
  className,
  saveButtonProps,
  publishButtonProps,
  ...stackProps
}: ActionButtonProps) {
  return (
    <Stack spacing={8} {...stackProps}>
      {article?.publishedAt ? (
        <Button {...publishButtonProps} type="submit" fullWidth>
          Save
        </Button>
      ) : (
        <>
          <Button {...saveButtonProps} type="submit" variant="default" fullWidth>
            Save Draft
          </Button>
          <Button {...publishButtonProps} type="submit" fullWidth>
            Publish
          </Button>
        </>
      )}
      {article?.publishedAt ? (
        <Text size="xs" color="dimmed">
          Published at {formatDate(article.publishedAt)}
        </Text>
      ) : (
        <Text size="xs" color="dimmed">
          Your article is currently{' '}
          <Tooltip label={hiddenLabel} {...tooltipProps}>
            <Text span underline>
              hidden
            </Text>
          </Tooltip>
        </Text>
      )}
    </Stack>
  );
}

type FormButtonProps = Pick<ButtonProps, 'disabled' | 'loading'> & {
  onClick: React.MouseEventHandler<HTMLButtonElement>;
};

type ActionButtonProps = StackProps & {
  saveButtonProps: FormButtonProps;
  publishButtonProps: FormButtonProps;
  article?: ArticleGetById;
};
