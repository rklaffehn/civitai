// src/pages/_app.tsx
import { ColorScheme, ColorSchemeProvider, MantineProvider } from '@mantine/core';
import { NotificationsProvider } from '@mantine/notifications';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { getCookie, getCookies, setCookie } from 'cookies-next';
import dayjs from 'dayjs';
import duration from 'dayjs/plugin/duration';
import isBetween from 'dayjs/plugin/isBetween';
import minMax from 'dayjs/plugin/minMax';
import relativeTime from 'dayjs/plugin/relativeTime';
import utc from 'dayjs/plugin/utc';
import type { NextPage } from 'next';
import type { AppContext, AppProps } from 'next/app';
import App from 'next/app';
import Head from 'next/head';
import type { Session } from 'next-auth';
import { SessionProvider, getSession } from 'next-auth/react';
import React, { ReactElement, ReactNode, useCallback, useEffect, useMemo, useState } from 'react';

import { AppLayout } from '~/components/AppLayout/AppLayout';
import { trpc } from '~/utils/trpc';
import '~/styles/globals.css';
import { CustomModalsProvider } from './../providers/CustomModalsProvider';
import { TosProvider } from '~/providers/TosProvider';
import { CookiesContext, CookiesProvider, parseCookies } from '~/providers/CookiesProvider';
import { MaintenanceMode } from '~/components/MaintenanceMode/MaintenanceMode';
// import { ImageProcessingProvider } from '~/components/ImageProcessing';
import { FeatureFlagsProvider } from '~/providers/FeatureFlagsProvider';
import { getFeatureFlags } from '~/server/services/feature-flags.service';
import type { FeatureAccess } from '~/server/services/feature-flags.service';
import { ClientHistoryStore } from '~/store/ClientHistoryStore';
import { isDev, isMaintenanceMode } from '~/env/other';
import { RegisterCatchNavigation } from '~/store/catch-navigation.store';
import { CivitaiLinkProvider } from '~/components/CivitaiLink/CivitaiLinkProvider';
import { MetaPWA } from '~/components/Meta/MetaPWA';
import PlausibleProvider from 'next-plausible';
import { CivitaiSessionProvider } from '~/components/CivitaiWrapped/CivitaiSessionProvider';
import { CookiesState, FiltersProvider, parseFilterCookies } from '~/providers/FiltersProvider';
import { RouterTransition } from '~/components/RouterTransition/RouterTransition';
import { HiddenPreferencesProvider } from '../providers/HiddenPreferencesProvider';
import { SignalProvider } from '~/components/Signals/SignalsProvider';
import { CivitaiPosthogProvider } from '~/hooks/usePostHog';
import { ReferralsProvider } from '~/components/Referrals/ReferralsProvider';
import { RoutedDialogProvider } from '~/components/Dialog/RoutedDialogProvider';
import { DialogProvider } from '~/components/Dialog/DialogProvider';
import { BrowserRouterProvider } from '~/components/BrowserRouter/BrowserRouterProvider';
import { IsClientProvider } from '~/providers/IsClientProvider';
import { StripeSetupSuccessProvider } from '~/providers/StripeProvider';
import { BaseLayout } from '~/components/AppLayout/BaseLayout';

dayjs.extend(duration);
dayjs.extend(isBetween);
dayjs.extend(minMax);
dayjs.extend(relativeTime);
dayjs.extend(utc);

type CustomNextPage = NextPage & {
  getLayout?: (page: ReactElement) => ReactNode;
  options?: Record<string, unknown>;
};

type CustomAppProps = {
  Component: CustomNextPage;
} & AppProps<{
  session: Session | null;
  colorScheme: ColorScheme;
  cookies: CookiesContext;
  filters: CookiesState;
  flags: FeatureAccess;
  isMaintenanceMode: boolean | undefined;
}>;

function MyApp(props: CustomAppProps) {
  const {
    Component,
    pageProps: {
      session,
      colorScheme: initialColorScheme,
      cookies,
      filters,
      flags,
      isMaintenanceMode,
      ...pageProps
    },
  } = props;
  const [colorScheme, setColorScheme] = useState<ColorScheme | undefined>(initialColorScheme);
  const toggleColorScheme = useCallback(
    (value?: ColorScheme) => {
      const nextColorScheme = value || (colorScheme === 'dark' ? 'light' : 'dark');
      setColorScheme(nextColorScheme);
      setCookie('mantine-color-scheme', nextColorScheme, {
        expires: dayjs().add(1, 'year').toDate(),
      });
    },
    [colorScheme]
  );

  useEffect(() => {
    if (colorScheme === undefined && typeof window !== 'undefined') {
      const osColor = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
      setColorScheme(osColor);
    }
  }, [colorScheme]);

  const getLayout = useMemo(
    () =>
      Component.getLayout ??
      ((page: React.ReactElement) => <AppLayout {...Component.options}>{page}</AppLayout>),
    [Component.getLayout, Component.options]
  );

  const content = isMaintenanceMode ? (
    <MaintenanceMode />
  ) : (
    <IsClientProvider>
      <ClientHistoryStore />
      <RegisterCatchNavigation />
      <RouterTransition />
      <SessionProvider session={session} refetchOnWindowFocus={false} refetchWhenOffline={false}>
        <FeatureFlagsProvider flags={flags}>
          <SignalProvider>
            <CivitaiSessionProvider>
              <CivitaiPosthogProvider>
                <CookiesProvider value={cookies}>
                  <ReferralsProvider>
                    <FiltersProvider value={filters}>
                      <HiddenPreferencesProvider>
                        <CivitaiLinkProvider>
                          <NotificationsProvider zIndex={9999}>
                            <BrowserRouterProvider>
                              <BaseLayout>
                                <CustomModalsProvider>
                                  <TosProvider>
                                    {getLayout(<Component {...pageProps} />)}
                                  </TosProvider>
                                  <StripeSetupSuccessProvider />
                                  <DialogProvider />
                                  <RoutedDialogProvider />
                                </CustomModalsProvider>
                              </BaseLayout>
                            </BrowserRouterProvider>
                          </NotificationsProvider>
                        </CivitaiLinkProvider>
                      </HiddenPreferencesProvider>
                    </FiltersProvider>
                  </ReferralsProvider>
                </CookiesProvider>
              </CivitaiPosthogProvider>
            </CivitaiSessionProvider>
          </SignalProvider>
        </FeatureFlagsProvider>
      </SessionProvider>
    </IsClientProvider>
  );

  return (
    <>
      <Head>
        <title>Civitai | Share your models</title>
        <MetaPWA />
      </Head>

      <ColorSchemeProvider
        colorScheme={colorScheme ?? 'dark'}
        toggleColorScheme={toggleColorScheme}
      >
        <MantineProvider
          theme={{
            colorScheme,
            components: {
              Modal: {
                styles: {
                  modal: { maxWidth: '100%' },
                },
                // defaultProps: {
                //   target:
                //     typeof window !== 'undefined' ? document.getElementById('root') : undefined,
                // },
              },
              Drawer: {
                styles: {
                  drawer: {
                    containerName: 'drawer',
                    containerType: 'inline-size',
                    display: 'flex',
                    flexDirection: 'column',
                  },
                  body: { flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' },
                  header: { margin: 0 },
                },
                // defaultProps: {
                //   target:
                //     typeof window !== 'undefined' ? document.getElementById('root') : undefined,
                // },
              },
              Tooltip: {
                defaultProps: { withArrow: true },
              },
              Popover: { styles: { dropdown: { maxWidth: '100vw' } } },
              Rating: { styles: { symbolBody: { cursor: 'pointer' } } },
              Switch: {
                styles: {
                  body: { verticalAlign: 'top' },
                  track: { cursor: 'pointer' },
                  label: { cursor: 'pointer' },
                },
              },
              Radio: {
                styles: {
                  radio: { cursor: 'pointer' },
                  label: { cursor: 'pointer' },
                },
              },
              Badge: {
                styles: { leftSection: { lineHeight: 1 } },
                defaultProps: { radius: 'sm' },
              },
              Checkbox: {
                styles: {
                  input: { cursor: 'pointer' },
                  label: { cursor: 'pointer' },
                },
              },
            },
            colors: {
              accent: [
                '#F4F0EA',
                '#E8DBCA',
                '#E2C8A9',
                '#E3B785',
                '#EBA95C',
                '#FC9C2D',
                '#E48C27',
                '#C37E2D',
                '#A27036',
                '#88643B',
              ],
              success: [
                '#9EC3B8',
                '#84BCAC',
                '#69BAA2',
                '#4CBD9C',
                '#32BE95',
                '#1EBD8E',
                '#299C7A',
                '#2F826A',
                '#326D5C',
                '#325D51',
              ],
            },
            black: '#222',
            other: {
              fadeIn: `opacity 200ms ease-in`,
            },
          }}
          withGlobalStyles
          withNormalizeCSS
        >
          <PlausibleProvider
            domain="civitai.com"
            customDomain="https://analytics.civitai.com"
            selfHosted
          >
            {content}
          </PlausibleProvider>
        </MantineProvider>
      </ColorSchemeProvider>
      {isDev && <ReactQueryDevtools />}
    </>
  );
}

MyApp.getInitialProps = async (appContext: AppContext) => {
  const initialProps = await App.getInitialProps(appContext);
  const url = appContext.ctx?.req?.url;
  const isClient = !url || url?.startsWith('/_next/data');

  const { pageProps, ...appProps } = initialProps;
  const colorScheme = getCookie('mantine-color-scheme', appContext.ctx) ?? 'dark';
  const cookies = getCookies(appContext.ctx);
  const parsedCookies = parseCookies(cookies);
  const filters = parseFilterCookies(cookies);

  if (isMaintenanceMode) {
    return {
      pageProps: {
        ...pageProps,
        colorScheme,
        cookies: parsedCookies,
        isMaintenanceMode,
        filters,
      },
      ...appProps,
    };
  } else {
    const hasAuthCookie =
      !isClient && Object.keys(cookies).some((x) => x.endsWith('civitai-token'));
    const session = hasAuthCookie ? await getSession(appContext.ctx) : null;
    const flags = getFeatureFlags({ user: session?.user });
    // Pass this via the request so we can use it in SSR
    if (session) {
      (appContext.ctx.req as any)['session'] = session;
      (appContext.ctx.req as any)['flags'] = flags;
    }
    return {
      pageProps: {
        ...pageProps,
        colorScheme,
        cookies: parsedCookies,
        session,
        flags,
        filters,
      },
      ...appProps,
    };
  }
};

export default trpc.withTRPC(MyApp);
