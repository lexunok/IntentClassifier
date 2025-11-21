namespace IntentClassifier
{
    /// <summary>
    /// Представляет один пример данных (одно предложение и его метка).
    /// </summary>
    public class Sample
    {
        public uint[] InputIds = [];
        public int Label;
    }
}
