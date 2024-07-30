$(document).ready(function() {
    $('#inputForm').on('submit', function(event) {
        event.preventDefault();
        var userInput = $('#userInput').val();
        var submitButton = $('#input-button');
        var originalText = submitButton.text();

        // Change button text to "Masking..."
        submitButton.text('Masking...');

        // Make AJAX call immediately after submission
        $.ajax({
            url: "/predict",
            method: "POST",
            data: { user_input: userInput },
            success: function(response) {
                $('#output-area').html(response.result);

                // Change button text to "Finished." after AJAX call completes
                submitButton.text('Finished.');

                // Revert button text to original after 2 seconds
                setTimeout(function() {
                    submitButton.text(originalText);
                }, 2000); // 2 seconds
            },
            error: function() {
                // Handle error case if needed
                submitButton.text('Error.');
                setTimeout(function() {
                    submitButton.text(originalText);
                }, 2000); // 2 seconds
            }
        });
    });

    // Clear button functionality
    $('#clear-button').on('click', function() {
        $('#output-area').html('');
    });

    // Copy button functionality
    $('#copy-button').on('click', function() {
        var outputText = $('#output-area').text();
        navigator.clipboard.writeText(outputText).then(function() {
            var copyButton = $('#copy-button');
            var originalText = copyButton.text();
            copyButton.text('Copied ✓');
            setTimeout(function() {
                copyButton.text(originalText);
            }, 2000); // Change text back to "Copy" after 2 seconds
        }).catch(function(err) {
            console.error('Could not copy text: ', err);
        });
    });
});
