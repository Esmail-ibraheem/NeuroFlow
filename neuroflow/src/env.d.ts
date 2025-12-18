  interface ImportMetaEnv {
    readonly VITE_API_KEY?: string;
    readonly VITE_AUTOTRAIN_API_URL?: string;
  }

  interface ImportMeta {
  readonly env: ImportMetaEnv;
}
