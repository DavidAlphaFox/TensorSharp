namespace TensorSharp.Server;

internal static class WebUiChatPolicy
{
    internal const string ModelSelectionLockedMessage =
        "Use /api/models/load to choose a model before chatting. Changing models during chat is not supported.";

    // 中文：校验 WebUI 聊天请求——未指定模型与后端时通过，否则拒绝并返回「聊天期间不可切换模型」的提示。
    public static bool TryValidateChatRequest(string requestedModel, string requestedBackend, out string error)
    {
        if (string.IsNullOrWhiteSpace(requestedModel) && string.IsNullOrWhiteSpace(requestedBackend))
        {
            error = null;
            return true;
        }

        error = ModelSelectionLockedMessage;
        return false;
    }
}

